import os
import requests


class HcpDownloader:

    def __init__(self, settings):
        self.settings = settings

    def load(self, path):

        if not os.path.isfile(path):
            #logging.info, file not found
            subject = path.split('/')[3]
            key = path.split('/MNINonLinear/')[1]
            url = self.settings['SERVERS']['hcp_server_url'].format(subject, subject, subject) + key
            #logging.info, "remote download from server + url"
            r = requests.get(url,
                             auth=(self.settings['CREDENTIALS']['hcp_server_username'], self.settings['CREDENTIALS']['hcp_server_password']),
                             stream=True)
            #log status code
            if r.status_code == 200:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, 'wb') as f:
                    #log.debug, "writing url to + path"
                    f.write(r.content)
                    #log.debug, "writing done"
            #else log.error depending on status code "request unsuccessful. Error + r.status_code"

        #else (invert it) logging.info(file is found)


class DtiDownloader:

    def __init__(self, settings):
        self.base_path = settings['SERVERS']['dti_server_url']

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
