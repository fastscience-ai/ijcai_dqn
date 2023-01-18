import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

novelty_introduced = 7
output_file_name ="DQN_novelty_adaptation.png"

def text_to_array(file):
    perf=[1.0 for i in range(novelty_introduced)] 
    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            perf.append(float(line.split(",")[0])/200.0)
        perf=np.asarray(perf)
    return perf

def plot_experiment_results(df, novelty_episode_number, title, plot_length=8):
    # df=df.drop(df[df.trial_type=='unknown'].index)
    plt.figure(figsize=(plot_length, 6))
    ax = sns.lineplot(data=df, y='performance', x='episode', ci=95, legend=None)
    ax.set(ylim=(0, 1.1))
    data_xmax = max(df['episode'])
    ax.set(xlim=(0,data_xmax))
    plt.axvline(x=novelty_episode_number, color='red')
    plt.title(title, fontsize=26)
    plt.xlabel("episodes", fontsize=26)
    plt.ylabel("performance", fontsize=26)

trial_data_prep = []
columns_prep = ['trial', 'episode', 'performance']

for i in range(10):
    arr = text_to_array("train_"+str(i+1)+".txt")
    for j  in  range(len(arr)):
        trial_data_prep.append([i+1, j+1, arr[j]])

trial_df = pd.DataFrame(trial_data_prep, columns=columns_prep)
plot_experiment_results(trial_df, novelty_introduced, "novelty: mass of cart x10")

# plt.legend()
#plt.show()
plt.savefig(output_file_name, bbox_inches="tight")

