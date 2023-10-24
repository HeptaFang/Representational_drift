import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import matplotlib.patches as mpatches


def main():
    cmaplist_detail = [(1.0, 1.0, 1.0), (0.2, 0.8, 0.2), (0.9, 0.9, 0.2), (0.8, 0.2, 0.2), (0.7, 0.2, 0.9)]
    cmaplist_binary = [(1.0, 0.3, 0.3), (0.2, 0.8, 0.2), (0.2, 0.8, 0.2), (0.2, 0.8, 0.2), (0.2, 0.8, 0.2)]
    legends = ['nan', 'high confidence', 'medium confidence', 'low confidence', 'nucleus']
    patches = mpatches.Patch(color='red', label='The red data')
    cmaplist = cmaplist_detail
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

        #     for t in range(session_num):
        #         conf_count = {1: 0, 2: 0, 3: 0, 4: 0, 'nan': 0, 'activity_all_nan': 0, 'activity_any_nan': 0}
        #         for i in range(4):
        #             conf_count[i + 1] = np.sum(confidence[t] == i + 1)
        #         conf_count['nan'] = np.sum(np.isnan(confidence[t]))
        #         conf_count['activity_all_nan'] = np.sum(np.all(np.isnan(activity_raw[t, :, 2:3]), axis=(0, 1)))
        #         conf_count['activity_any_nan'] = np.sum(np.any(np.isnan(activity_raw[t, :, 2:3]), axis=(0, 1)))
        #
        #         print(t, conf_count)
        #
        #     cell_available_day = np.sum(~np.isnan(confidence), 0)
        #     available_count = np.zeros(session_num + 1)
        #     for i in range(session_num + 1):
        #         available_count[i] = np.sum(cell_available_day == i)
        #
        #     print(available_count)
        #     # plt.plot(available_count[1:], '.-')
        #     # plt.show()
        #     plt.hist(cell_available_day, bins=session_num + 1, range=[0, session_num + 1])
        #     nonzero_max = np.max(available_count[1:])
        #     plt.title(f'mouse {mouse_i + 1}, {session_num} sessions')
        #     plt.xlabel('available time (day)')
        #     plt.ylabel('cell num')
        #     plt.ylim([0, nonzero_max * 1.1])
        #     plt.show()

        confidence[np.isnan(confidence)] = 0
        confidence_selected[np.isnan(confidence_selected)] = 0
        # plt.imsave(f'image\\confidence_mouse{mouse_i + 1}.bmp', confidence.T, cmap=cmap, vmin=0, vmax=4)
        # plt.imsave(f'image\\confidence_selected_mouse{mouse_i + 1}.bmp', confidence_selected.T, cmap=cmap, vmin=0, vmax=4)
        # plt.imshow(confidence[:, :100].T, cmap=cmap)
        # plt.imsave(f'image\\confidence_mouse{mouse_i + 1}.bmp', confidence[:, :100].T)
        # plt.xlabel('time (day)')
        # plt.ylabel('confidence level')
        # plt.colorbar()
        # plt.show()

        axs_original[mouse_i].imshow(confidence[:, :100].T, cmap=cmap, vmin=0, vmax=4)
        axs_original[mouse_i].set_title(f'mouse {mouse_i + 1} ({cell_num})')
        axs_selected[mouse_i].imshow(confidence_selected[:, :60].T, cmap=cmap, vmin=0, vmax=4)
        axs_selected[mouse_i].set_title(f'mouse {mouse_i + 1} ({selected_cell_num})')

    fig_original.supxlabel('session #')
    fig_original.supylabel('cell #')
    fig_original.suptitle('data validity')
    fig_original.savefig(f'image\\confidence_all.jpg')
    fig_selected.supxlabel('session #')
    fig_selected.supylabel('cell #')
    fig_selected.suptitle('data validity')
    fig_selected.savefig(f'image\\confidence_selected_all.jpg')


if __name__ == '__main__':
    main()
