from argparse import ArgumentParser
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help="Path to json file")
    parser.add_argument('-o', '--output', type=str, help="Path to output file")
    parser.add_argument('-s', '--stat', action='store_true', default=True)
    parser.add_argument('--min_thresh', type=float, default=0)
    parser.add_argument('--max_thresh', type=float, default=1)
    parser.add_argument('--acc', action='store_true', default=False)
    return parser.parse_args()

def main(args):
    input_file = open(args.input, 'r')
    output_file = open(args.output, 'w+')

    stat = {
        "class": [],
        "score": []
    }

    data = json.load(input_file)
    result = []
    for video in data:
        events = video['events']
        video_name = video['video']
        frames = []
        for event in events:
            event_class = event['label']
            if ('card' in event_class):
                frames.append(event)
            stat['class'].append(event_class)
            stat['score'].append(float(event['score']))
        if (len(frames)>0):
            result.append({
                'video': video_name,
                'events': frames
            })
    json.dump(result, output_file, indent=3)
    
    if (args.stat):
        stat_df = pd.DataFrame(stat)
        nrow=4
        fig, ax = plt.subplots(nrow, math.ceil(17/nrow), figsize=(20,12))
        i = 0
        for t in stat_df['class'].unique():
            sns.histplot(
                data = stat_df[stat_df['class'] == t], 
                x='score', 
                # hue='class',
                ax=ax[i%nrow][i//nrow],
                cumulative=args.acc
            )
            ax[i%nrow][i//nrow].set_title(t)
            ax[i%nrow][i//nrow].set_xlabel('')
            ax[i%nrow][i//nrow].set_xlim(args.min_thresh, args.max_thresh)
            i+=1
        plt.show()

if __name__ == "__main__":
    main(get_args())
