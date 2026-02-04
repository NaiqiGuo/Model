import os
import numpy as np
import matplotlib.pyplot as plt

def read_nf_file(path):
    """
    Read a natural frequencies csv that contains a single column of numbers.
    Returns a 1D numpy array of shape (n_modes,).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # loadtxt handles plain numeric files well
    arr = np.loadtxt(path, delimiter=",")
    arr = np.atleast_1d(arr).astype(float)

    # If it accidentally becomes 2D (n,1), flatten it
    return arr.reshape(-1)

def load_df_over_f_from_event_folders(base_dir, n_events=22,
                                      pre_name="pre_eq_natural_frequencies.csv",
                                      post_name="post_eq_natural_frequencies.csv"):
    """
    base_dir/
      1/pre_name, 1/post_name
      2/pre_name, 2/post_name
      ...
      n_events/pre_name, n_events/post_name

    Returns:
      event_ids: (n_events,)
      df_over_f: (n_events, n_modes)
      f_pre:     (n_events, n_modes)
      f_post:    (n_events, n_modes)
    """
    event_ids = np.arange(1, n_events + 1)

    f_pre_list = []
    f_post_list = []
    n_modes_ref = None

    for eid in event_ids:
        event_dir = os.path.join(base_dir, str(eid))
        pre_path  = os.path.join(event_dir, pre_name)
        post_path = os.path.join(event_dir, post_name)

        f_pre  = read_nf_file(pre_path)
        f_post = read_nf_file(post_path)

        if n_modes_ref is None:
            n_modes_ref = len(f_pre)
        else:
            if len(f_pre) != n_modes_ref or len(f_post) != n_modes_ref:
                raise ValueError(
                    f"Mode count mismatch at event {eid}: "
                    f"pre has {len(f_pre)}, post has {len(f_post)}, expected {n_modes_ref}"
                )

        f_pre_list.append(f_pre)
        f_post_list.append(f_post)

    f_pre  = np.vstack(f_pre_list)   # (n_events, n_modes)
    f_post = np.vstack(f_post_list)  # (n_events, n_modes)

    # Δf/f_pre
    df_over_f = (f_pre - f_post) / f_pre

    return event_ids, df_over_f, f_pre, f_post

def plot_mode_heatmap(event_ids, df_over_f, mode_index, out_png, title,
                      vmin=-0.3, vmax=0.3, cmap="Blues"):
    """
    mode_index: 0-based
    """
    mode_vals = df_over_f[:, mode_index]

    plt.figure(figsize=(12, 2.5))
    plt.imshow(mode_vals[np.newaxis, :],
               aspect="auto",
               cmap=cmap,
               vmin=vmin, vmax=vmax)
    plt.colorbar(label="Δf/f_pre")
    plt.xticks(range(len(event_ids)), event_ids, rotation=90)
    plt.yticks([0], [f"Mode {mode_index+1}"])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved: {out_png}")

# inelastic
base_dir_in = "/Users/guonaiqi/Documents/UCB/299/Example5-python/Model/frame/inelastic"
event_in, df_in, fpre_in, fpost_in = load_df_over_f_from_event_folders(base_dir_in, n_events=22)

plot_mode_heatmap(
    event_in, df_in, mode_index=0,
    out_png="heatmap_inelastic_mode1.png",
    title="Inelastic Model. Mode 1 Natural Freq Change"
)


# base_dir_el = "/Users/guonaiqi/Documents/UCB/299/Example5-python/Model/frame/elastic"
# event_el, df_el, fpre_el, fpost_el = load_df_over_f_from_event_folders(base_dir_el, n_events=22)
# plot_mode_heatmap(event_el, df_el, 0, "heatmap_elastic_mode1.png", "Elastic Model. Mode 1 Natural Freq Change")
