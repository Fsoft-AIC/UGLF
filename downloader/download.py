import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
from argparse import ArgumentParser

def get_args():
    parser = ArgumentParser()
    parser.add_argument('-p', '--password', type=str, default=None)
    parser.add_argument('-d', '--directory', type=str, required=True)
    parser.add_argument('-l', '--label', action='store_true')
    parser.add_argument('-b', '--baidu', action='store_true')
    parser.add_argument('-hq', '--high_quality', action='store_true')
    parser.add_argument('-lq', '--low_quality', action='store_true')
    return parser.parse_args()

def main(args):
    print(f"Password: {args.password}")
    print(f"Directory: {args.directory}")

    mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory=args.directory)
    mySoccerNetDownloader.password = args.password

    if (args.label):
        print("Download label...")
        mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train","valid","test"])

    if (args.baidu):
        print("Download Baidu feature...")
        mySoccerNetDownloader.downloadGames(files=["1_baidu_soccer_embeddings.npy", "2_baidu_soccer_embeddings.npy"], split=["train","valid","test","challenge"])

    if (args.high_quality):
        print("Download HQ version...")
        mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv", "video.ini"], split=["test"])

    if (args.low_quality):
        print("Download LQ version...")
        mySoccerNetDownloader.downloadGames(files=["1.mkv", "2.mkv"], split=["train","valid","test","challenge"])

if __name__=="__main__":
    main(get_args())
