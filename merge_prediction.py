import json
from argparse import ArgumentParser

classes = [
    "Penalty",
    "Kick-off",
    "Goal",
    "Substitution",
    "Offside",
    "Shots on target",
    "Shots off target",
    "Clearance",
    "Ball out of play",
    "Throw-in",
    "Foul",
    "Indirect free-kick",
    "Direct free-kick",
    "Corner",
    "Yellow card",
    "Red card",
    "Yellow->red card"
]

def get_args():
    parser = ArgumentParser()
    
    parser.add_argument("first_prediction_dir", type=str)
    parser.add_argument("second_prediction_dir", type=str)
    parser.add_argument("output_dir", type=str)

    splitting_fn = lambda s: [item for item in s.split(',')]
    parser.add_argument("--either", default=[], type=splitting_fn)
    parser.add_argument("--both", default=[], type=splitting_fn)
    parser.add_argument("--first", default=[], type=splitting_fn)
    parser.add_argument("--second", default=[], type=splitting_fn)

    return parser.parse_args()

def load_file(file1Path, file2Path):
    with open(file1Path, 'r+') as inFile:
        jsonObj1 = json.load(inFile)
    with open(file2Path, 'r+') as inFile:
        jsonObj2 = json.load(inFile)
    return (jsonObj1, jsonObj2)

def assert_class(eitherClazz, bothClazz, file1Clazz, file2Clazz):
    global classes
    for clazz in bothClazz + file1Clazz + file2Clazz + eitherClazz:
        assert clazz in classes, f"{clazz} must in {classes}"

def output_result(outPath, new_result):
    with open(outPath, 'w+') as outFile:
        json.dump(new_result, outFile)

def main(args):
    global classes
    
    file1Path = args.first_prediction_dir
    file2Path = args.second_prediction_dir
    outPath = args.output_dir

    eitherClazz = args.either
    bothClazz = args.both
    file1Clazz = args.first
    file2Clazz = args.second
    print("First prediction path:", file1Path)
    print("Second prediction path:", file2Path)
    print("Output path:", outPath)

    print("Loading file...")
    jsonObj1, jsonObj2 = load_file(file1Path, file2Path)
    
    bothClazz = list(set(bothClazz) - set(eitherClazz))
    file2Clazz = list(set(file2Clazz) - set(bothClazz) - set(eitherClazz))
    file1Clazz = list(set(classes) - set(file2Clazz) - set(bothClazz) - set(eitherClazz))
    assert_class(eitherClazz, bothClazz, file1Clazz, file2Clazz)
    print("Either class:", eitherClazz)
    print("Both class:", bothClazz)
    print("First class:", file1Clazz)
    print("Second class:", file2Clazz)

    new_result = []
    for games1, games2 in zip(jsonObj1, jsonObj2):
        assert games1['video']==games2['video'], "The video names of 2 predictions are not match"

        events1 = games1['events']
        events2 = games2['events']
        video = games1['video']
        fps = games1['fps']
        new_events = []

        map_frame = {}

        for event in events1:
            label = event['label']
            frame = event['frame']
            if (frame not in map_frame):
                map_frame[frame] = []
            map_frame[frame].append(label)
            if (label in eitherClazz):
                new_events.append(event)
            if (label in file1Clazz):
                new_events.append(event)

        for event in events2:
            label = event['label']
            frame = event['frame']
            if ((frame in map_frame) and (label in map_frame[frame])):
                if (label in bothClazz):
                    new_events.append(event)
            if (label in file2Clazz):
                new_events.append(event)
            if (label in eitherClazz):
                new_events.append(event)

        new_game = {
            'video': video,
            'events': new_events,
            'fps': fps
        }
        for event in new_events:
            label = event['label']
        new_result.append(new_game)
    output_result(outPath, new_result)

main(get_args())
