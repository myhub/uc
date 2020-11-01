# coding: utf-8
from __future__ import print_function, division, unicode_literals, absolute_import
import numpy
import warnings
import sys
import matplotlib.pyplot as plt


def plot_fig_(title, show, fname, zoom, dpi):
    if title is not None:
        plt.title(title)

    fig = plt.gcf()
    plt.rc('font', size=14)
    if zoom is not None:
        if not isinstance(zoom, (tuple, list)):
            zoom = (zoom, zoom)
        size_inches = fig.get_size_inches()
        fig.set_size_inches(
            size_inches[0] * zoom[0], size_inches[1] * zoom[1])

    if fname is not None:
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    elif show is None:
        show = True

    if show:
        plt.show()

    plt.close(fig)


def plot_image(
        data,
        expand=False,

        title=None,
        show=None,
        fname=None,
        zoom=None,
        dpi=100
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            arr = numpy.array(data, numpy.float64)

            shape = arr.shape
            if len(shape) > 1:

                img = (arr - arr.min()) / (arr.max() - arr.min())

                if len(shape) == 2:
                    plt.imshow(img, cmap='gray')
                elif len(shape) == 3:
                    if expand:
                        if shape[0] > shape[2]:
                            img = img.transpose(1, 2, 0)

                        img = img.reshape(1, -1, img.shape[2])
                        plt.imshow(img[0], cmap='gray')
                    else:
                        if shape[0] < shape[2]:
                            img = img.transpose(1, 2, 0)

                        if img.shape[2] <= 2:
                            plt.imshow(img[:, :, 0], cmap='gray')
                        else:
                            plt.imshow(img[:, :, :4])

                plt.grid(False)

                plot_fig_(title=title, show=show,
                          fname=fname, zoom=zoom, dpi=dpi)

        except:
            print("plot_image error", file=sys.stderr)


def plot_importance(
    feature_importances,
    max_num_features=64,
    feature_names=None,

    title='Feature Importances',
    show=None,
    fname=None,
    zoom=None,
    dpi=100
):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            import pandas

            arr = numpy.array(feature_importances, numpy.float64).reshape(-1)

            feature_names_ = ['f'+str(i) for i in range(len(arr))]

            for i in range(len(feature_importances)):
                if feature_names is not None and i < len(feature_names):
                    feature_names_[i] = feature_names[i]

            plt.rc('font', **{'family': 'SimHei'})

            (pandas.Series(arr, index=feature_names_)
                .nlargest(max_num_features).sort_values()
                .plot(kind='barh'))

            plot_fig_(title=title, show=show,
                      fname=fname, zoom=zoom, dpi=dpi)

        except:
            print("plot_importance error", file=sys.stderr)


if __name__ == '__main__':
    plot_image([[1, 2], [3, 4]], fname='f1', zoom=0.5, show=True)
    plot_importance([[1, 2], [3, 4]], feature_names=['-啊aaaaaaaaaaaaaaa'])

    importance = list(range(1000))
    feature_names = ['X'*20 + str(i) for i in range(1000)]
    plot_importance(importance, 64, feature_names=feature_names,
                    fname='f3', show=True, zoom=[1.5, 4])
