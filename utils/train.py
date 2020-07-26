import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook

inp_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_gallery(images, h, w, n_row=3, n_col=6):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.5 * n_col, 1.7 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        inp = images[i].reshape(64, 64, 3)
        plt.imshow(inp, cmap=plt.cm.gray, vmin=-1, vmax=1, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.show()


def onehot(x):
    result_tensor = torch.zeros(x.shape[0], 2)

    for i in range(0, x.shape[0]):
        if x[i] >= 0:
            result_tensor[i] = torch.Tensor([1, 0])
        else:
            result_tensor[i] = torch.Tensor([0, 1])

    return result_tensor


def draw_graph(train_hist, test_hist):
    """
    Вспомогательная функция для обучения, рисует график лосса
    """
    plt.figure(figsize=(12, 7))
    plt.subplot(1, 2, 1)
    plt.plot(train_hist, label="train")
    plt.title('train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(test_hist)
    plt.title('val loss')
    plt.show()


def do_epoch(model, loss_func, data, optimizer=None, name=None, cvae=False):
    epoch_loss = []
    plot_dir = []

    is_train = not optimizer is None

    name = name or ''
    model.train(is_train)

    batches_count = len(data)

    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batches_count) as progress_bar:
            for dt in data:
                if cvae:
                    inputs = dt[0].view(-1, inp_size * inp_size * 3).to(device).float()
                    labels = onehot(dt[1]).to(device)
                    faces = torch.cat((inputs, labels), dim=-1)
                    model_output = model(faces, labels)

                else:
                    inputs = dt.view(-1, inp_size * inp_size * 3).to(device).float()
                    model_output = model(inputs)

                loss = loss_func(inputs, model_output)

                if optimizer:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()

                epoch_loss.append(loss.item())

                progress_bar.update()
                progress_bar.set_description('{:>5s} Loss = {:.5f}'.format(
                    name, loss.item()))

            progress_bar.set_description('{:>5s} Average Loss = {:.5f}'.format(
                name, torch.mean(torch.FloatTensor(epoch_loss))))

            if not optimizer:
                if not cvae:
                    plot_dir.append(inputs.detach().cpu().view(-1, 3, inp_size, inp_size).numpy())
                    plot_dir.append(model(inputs.float())[0].detach().cpu().view(-1, 3, inp_size, inp_size).numpy())
                else:
                    plot_dir.append(dt[0].detach().cpu().view(-1, 3, inp_size, inp_size).numpy())
                    plot_dir.append(
                        model(faces.float(), labels)[0].detach().cpu().numpy().reshape(-1, inp_size, inp_size, 3))

    return plot_dir, epoch_loss


def fit(model, loss, optimizer, train_data, val_data=None, epochs_count=1, cvae=False, show_img_val=20):
    train_all_loss = []
    val_all_loss = []

    for epoch in range(epochs_count):
        name_prefix = '[{} / {}] '.format(epoch + 1, epochs_count)
        _, train_loss = do_epoch(model, loss, train_data, optimizer,
                                 name_prefix + 'Train:', cvae)
        train_all_loss += train_loss

        if not val_data is None:
            out, val_loss = do_epoch(model, loss, val_data, None,
                                     name_prefix + '  Val:', cvae)
            x_val, reconstruction = out
            val_all_loss += val_loss

            if epoch % show_img_val == 0:
                plot_gallery(x_val, inp_size, inp_size, n_row=1, n_col=5)
                plot_gallery(reconstruction, inp_size, inp_size, n_row=1, n_col=5)

    draw_graph(train_all_loss, val_all_loss)
