import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.patches as mpatches


def main():
    # cmaplist_binary = [(0.8, 0.8, 0.8), (0.2, 0.2, 1.0)]
    cmaplist_binary = [(1.0, 1.0, 1.0), (0.2, 0.2, 1.0)]
    # legends = ['nan', 'available']
    # patches = [mpatches.Patch(color=cmaplist_binary[i], label=legends[i]) for i in range(2)]
    cmaplist = cmaplist_binary
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, len(cmaplist))

    # figure properties
    fig_original = plt.figure(figsize=(10, 6), dpi=300, layout='constrained')
    spacing = 0
    axs_original = fig_original.subplot_mosaic([[0, 1, 2, 3, 4]],
                                               width_ratios=[59 + spacing, 23 + spacing, 42 + spacing, 25 + spacing,
                                                             33 + spacing],
                                               sharey=True)
    fig_selected = plt.figure(figsize=(10, 6), dpi=300, layout='constrained')
    axs_selected = fig_selected.subplots(1, 5, sharey=True)

    for mouse_i in range(5):
        activity_raw = np.load(f'dataset\\mouse{mouse_i + 1}_activity_raw.npy')
        confidence = np.load(f'dataset\\mouse{mouse_i + 1}_confidence.npy')
        activity_selected_raw = np.load(f'dataset\\mouse{mouse_i + 1}_activity_selected_raw.npy')
        confidence_selected = np.load(f'dataset\\mouse{mouse_i + 1}_confidence_selected.npy')
        session_num, cell_num = confidence.shape
        session_num, selected_cell_num = confidence_selected.shape

        print(activity_raw.shape)
        print(activity_selected_raw.shape)
        availability = ~np.all(np.isnan(activity_raw[:, :, 1:2, :]), axis=(1, 2))
        availability_selected = ~np.all(np.isnan(activity_selected_raw[:, :, 1:2, :]), axis=(1, 2))

        # confidence[np.isnan(confidence)] = 0
        # plt.imsave(f'image\\availability_mouse{mouse_i + 1}.bmp', availability.T, cmap=cmap)
        # plt.imsave(f'image\\availability_selected_mouse{mouse_i + 1}.bmp', availability_selected.T, cmap=cmap)
        # plt.imshow(availability[:, :100].T, cmap=cmap)
        # plt.imsave(f'image\\confidence_mouse{mouse_i + 1}.bmp', confidence[:, :100].T)
        # plt.xlabel('time (day)')
        # plt.ylabel('confidence level')
        # plt.colorbar()
        # plt.show()

        axs_original[mouse_i].imshow(availability[:, :100].T, cmap=cmap, vmin=0, vmax=1)
        axs_original[mouse_i].set_title(f'mouse {mouse_i + 1} ({cell_num})')
        axs_selected[mouse_i].imshow(availability_selected[:, :60].T, cmap=cmap, vmin=0, vmax=1)
        axs_selected[mouse_i].set_title(f'mouse {mouse_i + 1} ({selected_cell_num})')

    fig_original.supxlabel('session #')
    fig_original.supylabel('cell #')
    fig_original.suptitle('data validity')
    fig_original.savefig(f'image\\validity_all.jpg')
    fig_selected.supxlabel('session #')
    fig_selected.supylabel('cell #')
    fig_selected.suptitle('data validity')
    fig_selected.savefig(f'image\\validity_selected_all.jpg')


if __name__ == '__main__':
    main()
