import matplotlib.pyplot as plt

def plotScore(plotPath,plot_x,plot_loss,plot_acc):
    plt.figure(figsize=(10, 10), dpi=100)
    grid = plt.GridSpec(3, 2, wspace=0.2, hspace=0.5)
    plt.subplot(grid[0, 0])
    plt.plot(plot_x, plot_acc, 'o-b')
    plt.title('accurary', fontsize=20)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('accurary', fontsize=14)

    plt.subplot(grid[0, 1])
    plt.plot(plot_x, plot_loss, 'o-r')
    plt.title('loss', fontsize=20)
    plt.ylim(0, 1)
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.savefig(fname=plotPath+'/score.svg', format='svg')
    #plt.show()

def plotScoreLoss(plotPath, x_train_loss, y_train_loss):
    plt.figure()
    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')  # x轴标签
    plt.ylabel('loss')  # y轴标签

    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
    #plt.legend()
    plt.title('Loss curve')
    #plt.show()
    plt.savefig(fname=plotPath + '/loss.svg', format='svg')

def plotScoreDice(plotPath, x_train_loss, y_train_loss):
    plt.figure()
    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')  # x轴标签
    plt.ylabel('accuracy')  # y轴标签

    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="val accuracy")
    #plt.legend()
    plt.title('accuracy curve')
    #plt.show()
    plt.savefig(fname=plotPath + '/accuracy.svg', format='svg')
