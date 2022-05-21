import os
import json
import glob
import pickle
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset

from utils.utils import print_log
from src.nltk import sentence_get_pos_tags


class oriDataset:
    def __init__(self, mode, imdb_path, data_path):
        self.IMDB_PATH = imdb_path
        self.DATA_PATH = data_path
        assert mode in ["train", "test", "total"]

        self.imdb_pos_path = os.path.join(self.IMDB_PATH, mode, "pos")
        self.imdb_neg_path = os.path.join(self.IMDB_PATH, mode, "neg")

        self.imdb_pos_urls = os.path.join(self.IMDB_PATH, mode, "urls_pos.txt")
        self.imdb_neg_urls = os.path.join(self.IMDB_PATH, mode, "urls_neg.txt")

        self.json_data = json.load(open(os.path.join(self.DATA_PATH, mode + ".json")))

        self.imdb_data = list()                     # list of reviews
        self.imdb_ids = list()                      # index to imdbID
        self.num_of_genres = 0                      # number of genres
        self.dict_of_genres = dict()                # genre to genreID
        self.genreID_to_imdbID = defaultdict(list)  # genreID to imdbID
        self.imdbID_to_data = defaultdict(list)     # imdbID to reviews

        self.init()

    def init(self):
        # create dict_of_genres
        if os.path.isfile("dict_of_genres.json"):
            self.dict_of_genres = json.load(open("dict_of_genres.json"))
        else:
            set_of_genres = set()
            for imdbID, data in self.json_data.items():
                genres = [genre.strip() for genre in data["Genre"].split(", ")]
                set_of_genres.update(set(genres))
            for gid, genre in enumerate(set_of_genres):
                self.dict_of_genres[genre] = gid
        self.num_of_genres = len(self.dict_of_genres)

        # create gener_to_imdbID
        for imdbID, data in self.json_data.items():
            genres = [genre.strip() for genre in data["Genre"].split(", ")]
            for genre in genres:
                genreID = self.dict_of_genres[genre]
                self.genreID_to_imdbID[genreID].append(imdbID)

        # load positive reviews
        pos_urls_fp = open(self.imdb_pos_urls, "r")
        filenames = glob.glob(os.path.join(self.imdb_pos_path, "*.txt"))
        for filename in sorted(filenames, key=lambda filename: \
                int(os.path.basename(filename).split('_')[0])):
            imdbID = pos_urls_fp.readline().split('/')[-2]
            with open(filename, "r") as fp:
                data = fp.readline()
            fp.close()
            self.imdb_ids.append(imdbID)
            self.imdbID_to_data[imdbID].append(data)
            self.imdb_data.append(data)

        # load negative reviews
        neg_urls_fp = open(self.imdb_neg_urls, "r")
        filenames = glob.glob(os.path.join(self.imdb_neg_path, "*.txt"))
        for filename in sorted(filenames, key=lambda filename: \
                int(os.path.basename(filename).split('_')[0])):
            imdbID = neg_urls_fp.readline().split('/')[-2]
            with open(filename, "r") as fp:
                data = fp.readline()
            fp.close()
            self.imdb_ids.append(imdbID)
            self.imdbID_to_data[imdbID].append(data)
            self.imdb_data.append(data)

    def get_data_from_genres(self, list_of_genres):
        out = []
        for genres in list_of_genres:
            out.append([])
            for genreID in genres:
                imdbIDs = self.genreID_to_imdbID[genreID]
                for imdbID in imdbIDs:
                    out[-1].extend(self.imdbID_to_data[imdbID])
        return out

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        dic = {
            "index": index,
            "data": self.imdb_data[index],
            "imdbID": self.imdb_ids[index],
            "info": self.preprocess(self.json_data[self.imdb_ids[index]])
        }

        return dic

    def preprocess(self, label):
        one_hot = [0 for _ in range(self.num_of_genres)]
        for genre in label["Genre"].split(", "):
            one_hot[self.dict_of_genres[genre]] = 1

        label["Genre_one_hot"] = one_hot
        return label


class iMDBDataset(Dataset):
    def __init__(self, mode, imdb_path, data_path):
        self.IMDB_PATH = imdb_path
        self.DATA_PATH = data_path
        assert mode in ["train", "test", "total"]

        self.imdb_pos_path = os.path.join(self.IMDB_PATH, mode, "pos")
        self.imdb_neg_path = os.path.join(self.IMDB_PATH, mode, "neg")

        self.imdb_pos_urls = os.path.join(self.IMDB_PATH, mode, "urls_pos.txt")
        self.imdb_neg_urls = os.path.join(self.IMDB_PATH, mode, "urls_neg.txt")

        self.json_data = json.load(open(
            os.path.join(self.DATA_PATH, "{}.json".format(mode))))

        self.imdb_data = []
        self.imdb_ids = []
        self.num_of_genres = 0
        self.dict_of_genres = {}

        self.init()

    def init(self):
        set_of_genres = set()
        for _, v in self.json_data.items():
            if "Genre" in v:
                genres = [genre.strip() for genre in v["Genre"].split(", ")]
                set_of_genres.update(set(genres))

        self.num_of_genres = len(set_of_genres)
        for i, genre in enumerate(set_of_genres):
            self.dict_of_genres[genre] = i

        print("Load positive data...")
        pos_urls_fp = open(self.imdb_pos_urls, "r")
        for i, filename in enumerate(sorted(
            glob.glob(os.path.join(self.imdb_pos_path, "*.txt")))):
            self.imdb_ids.append(pos_urls_fp.readline().split('/')[-2])
            with open(filename, "r") as fp:
                self.imdb_data.append(fp.readline())
            fp.close()

        print("Load negative data...")
        neg_urls_fp = open(self.imdb_neg_urls, "r")
        for i, filename in enumerate(sorted(
            glob.glob(os.path.join(self.imdb_neg_path, "*.txt")))):
            self.imdb_ids.append(neg_urls_fp.readline().split('/')[-2])
            with open(filename, "r") as fp:
                self.imdb_data.append(fp.readline())
            fp.close()
        print("done, ", len(self.json_data))

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        id = list(self.json_data.keys())[index]
        if id not in self.imdb_ids:
            return self.__getitem__(index + 1)
        while id not in self.imdb_ids:
            import random
            index = random.randint(0, len(self.json_data) - 1)
            id = list(self.json_data.keys())[index]

        ids = sorted(self.imdb_ids).index(id)
        dic = {
            "index": ids,
            "data": self.imdb_data[ids],
            "imdbID": self.imdb_ids[ids],
            "labels": self.preprocess(self.json_data[id])
        }

        return dic

    def preprocess(self, label):
        one_hot = [0 for _ in range(self.num_of_genres)]
        if "Genre" in label:
            for genre in label["Genre"].split(", "):
                one_hot[self.dict_of_genres[genre]] = 1

        label = np.array(one_hot)
        return label


def get_dataset_and_pos(cfg):
    gallery_set = oriDataset(cfg.set_name, imdb_path=cfg.imdb_path,
                             data_path=cfg.data_path)
    if cfg.re_pos_tags or not os.path.exists(cfg.pos_dataset):
        print_log("Generating POS tags of dataset")
        pos_dataset = {}
        pbar = [sentence for sentences in gallery_set.imdbID_to_data.values()
                for sentence in sentences if sentence]
        for sentence in pbar:
            pos_dataset[sentence] = sentence_get_pos_tags(sentence)
        with open(cfg.pos_dataset, 'wb') as f:
            pickle.dump(pos_dataset, f)
    else:
        with open(cfg.pos_dataset, 'rb') as f:
            pos_dataset = pickle.load(f)

    return gallery_set, pos_dataset
