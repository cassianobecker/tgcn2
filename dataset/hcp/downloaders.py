import os
import requests


class HCPDownloader:

    def __init__(self, settings):
        self.settings = settings

    def load(self, path):

        if not os.path.isfile(path):

            subject = path.split('/')[3]
            key = path.split('/MNINonLinear/')[1]
            url = self.settings['DIRECTORIES']['HCPDir'].format(subject, subject, subject) + key
            r = requests.get(url,
                             auth=(self.settings['CREDENTIALS']['Username'], self.settings['CREDENTIALS']['Password']),
                             stream=True)
            if r.status_code == 200:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as f:
                    f.write(r.content)


class GitDownloader:

    def __init__(self, settings):
        self.base_path = settings['DIRECTORIES']['GitDir']

    def load(self, path):

        if not os.path.isfile(path):

            subject = path.split('/')[3]
            key = path.split('/MNINonLinear/')[1]
            temp = key.split('/', 1)
            key = temp[0] + '/' + subject + '/' + temp[1]
            url = self.base_path + key
            r = requests.get(url)

            if r.status_code == 200:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as f:
                    f.write(r.content)
