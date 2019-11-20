import csv
import pandas as pd
from collections import namedtuple, defaultdict
from datetime import datetime
import numpy as np
import textwrap
import random
from scipy.special import binom
from tqdm import tqdm
import pickle
import textwrap
from dataclasses import dataclass
import time

#########################
# Commenti sul Dataset
#########################
# i campi index e index_0 sono sempre uguali
# gli indici sono univoci (identificano i record). Sono interi non consecutivi
# Cosa rappresenta l'ultimo valore di seq_uu?
# uu contiene le differenze di tu
# dd contiene le differenze di td
# du contiene le differenze tu - td (per quanto tempo viene premuto un tasto)
# ud contiene le differenze td[n+1] - tu[n] (tempo tra il rilascio di un tasto e la pressione del succesivo)

pd.set_option('display.max_rows', None)

# namedtuple che rapprsenta un record del dataset
Record = namedtuple(
    "Record", "index_0 index expected_result user_id timestamp seq_tu seq_td seq_uu seq_du seq_ud seq_dd")

# i 6 campi
fields = 'seq_tu seq_td seq_uu seq_du seq_ud seq_dd'.split()
############
# Costanti
############
POSITIVE_RECORD = 1
NEGATIVE_RECORD = -1


class DataSet:
    '''Rappresenta il Dataset'''

    def __init__(self, file_path_or_list_of_records, name=None, info=None):
        '''
        Args
        ----
        file_path_or_list_of_records(str | list) : il path completo del file del dataset oppure
                                                   una lista di record
        name (str): il nome del Dataset (es. Training Set o Test Set)
        info (str): ulteriori informazioni sul dataset
        '''
        self.name = name
        self.info = info

        if type(file_path_or_list_of_records) == str:
            self.file_path = file_path_or_list_of_records
            self.records = self.load()
        elif type(file_path_or_list_of_records) == list:
            self.file_path = None
            self.records = file_path_or_list_of_records
        else:
            raise ValueError("file_path_or_list_of_records deve essere una lista di record "
                             "o una stringa")

        # elimina i record con expected_result diverso da 1 o -1
        self.delete_records_with_wrong_result()
        # verifica che gli index siano tutti differenti
        assert len({record.index for record in self.records}) == len(self)

    def load(self):
        ''' Carica il dataset'''
        records = []
        with open(self.file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            # salta la prima riga
            next(csv_reader)
            for record in csv_reader:
                index_0 = int(record[0])
                index = int(record[1])
                expected_result = int(record[2])
                user_id = record[3]
                timestamp = datetime.fromisoformat(record[4])
                seq_tu = eval(record[5])
                seq_td = eval(record[6])
                seq_uu = eval(record[7])
                seq_du = eval(record[8])
                seq_ud = eval(record[9])
                seq_dd = eval(record[10])
                records.append(Record(index_0, index, expected_result, user_id,
                                      timestamp, seq_tu, seq_td, seq_uu, seq_du, seq_ud, seq_dd))
        return records

    def delete_records_with_wrong_result(self):
        self.cleaned_records = [
            record for record in self.records if record.expected_result in [-1, 1]]
        self.deleted_records = len(self.records) - len(self.cleaned_records)
        self.records = self.cleaned_records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        ''' Ritorna una namedtuple Record'''
        return self.records[idx]

    def get_positive_records_number(self):
        ''' Numero totale di positivi'''
        return len([record for record in self.records if record.expected_result == POSITIVE_RECORD])

    def get_negative_records_number(self):
        ''' Numero totale di negativi'''
        return len([record for record in self.records if record.expected_result == NEGATIVE_RECORD])

    def get_user_ids(self):
        ''' Restituisce una lista con gli tutti gli id utente distinti presenti nel dataset

        Ret
        ---
        la lista di tutti gli id utente (list)
        '''
        return list({record.user_id for record in self.records})

    def get_field_from_sample(self, idx, field):
        return getattr(self[idx], field)

    def get_all_fields_from_sample(self, idx):
        all_fields = []
        for field in fields:
            all_fields += getattr(self[idx], field)
        return all_fields

    def get_custom_sample(self, f, idx):
        return f(self[idx])

    def apply_tranformer(self, transformer):
        ''' Applica una trasformazione a ciascun record. Utile per normalizzare

        Args
        ----
        transformer: è un oggetto che implementa transform(che prende un record e 
        restituisce un record) che viene applicata ad ogni record del dataset
        '''
        for i, record in enumerate(self.records):
            self.records[i] = transformer.transform(record)

    def get_records_with_id(self, user_id):
        ''' Ritorna i record associati ad uno user_id

        Args
        ----
        user_id (str) : unouser id

        Ret
        ---
        tuple(positive_records, negative_records)
        positive_records (list): lista con i record positivi
        negative_records (list): lista con i record negativi (fraudolenti)
        '''
        positive_records = [record for record in self.records if record.expected_result ==
                            POSITIVE_RECORD and record.user_id == user_id]
        negative_records = [record for record in self.records if record.expected_result ==
                            NEGATIVE_RECORD and record.user_id == user_id]
        return positive_records, negative_records

    def get_all_users_info(self):
        ''' Restituisce info su tutti gli user_id
        Ret
        ---
        Una lista di tuple dove
        tuple[0] è uno user_id
        tuple[1] è il numero di record positivi
        tuple[2] è il numero di record negativi
        '''
        users_info = []
        for user_id in self.get_user_ids():
            positive_records, negative_records = self.get_records_with_id(
                user_id)
            users_info.append(
                (user_id, len(positive_records), len(negative_records)))
        return(users_info)

    def print_dataset_info(self):
        ''' Visualizza informazioni sul dataset'''

        user_ids, positive_record_num, negative_record_num = list(
            zip(*self.get_all_users_info()))
        df = pd.DataFrame({
            "user_id": user_ids,
            "positive_record_num": positive_record_num,
            "negative_record_num": negative_record_num})

        info_string = f'''\
            Nome del Dataset = {self.name}
            Informazioni = {self.info}
            Nome del file = {self.file_path}
            Numero di record = {len(self)}
            Record cancellati = {self.deleted_records}
            Informazioni utenti:'''

        print(textwrap.dedent(info_string))
        print(df)

    def get_all_indexes(self):
        ''' Ritorna una lista con gli indici (campo index, sono interi)'''
        return [record.index for record in self.records]

    def print_record(self, record):
        ''' Stampa a video un record (namedtuple Record)'''
        record_info_string = f'''\
            index = {record.index}
            expected_result = {record.expected_result}
            user_id = {record.user_id}
            timestamp = {record.timestamp}
            '''
        print(textwrap.dedent(record_info_string))

        df = pd.DataFrame({
            'seq_tu': record.seq_tu,
            'seq_td': record.seq_td,
            'seq_uu': record.seq_uu,
            'seq_du': record.seq_du,
            'seq_ud': record.seq_ud,
            'seq_dd': record.seq_dd
        })
        print(df)


def train_dev_test_split_by_user(dataset, train_user_ids, dev_user_ids, test_user_ids):
    ''' divide il dataset in train, dev e test set

    Args
    ----
    dataset (Dataset): dataset da dividere
    train_user_ids (list(str)): lista di utenti da inserire nel training set
    dev_user_ids: idem
    test_user_ids: idem

    Ret
    ---
    una tupla con 3 elementi
    training_set (Dataset): il training set
    dev_set: idem
    test_set: idem
    '''

    training_set_list = []
    test_set_list = []
    dev_set_list = []

    # training set
    for user_id in train_user_ids:
        positive_records, negative_records = dataset.get_records_with_id(
            user_id)
        training_set_list += positive_records + negative_records
        training_set = DataSet(
            training_set_list, name="Training Set")

    # dev set
    for user_id in dev_user_ids:
        positive_records, negative_records = dataset.get_records_with_id(
            user_id)
        dev_set_list += positive_records + negative_records
        dev_set = DataSet(dev_set_list, name="Dev Set")

    # test set
    for user_id in test_user_ids:
        positive_records, negative_records = dataset.get_records_with_id(
            user_id)
        test_set_list += positive_records + negative_records
        test_set = DataSet(test_set_list, name="Test Set")

    return training_set, dev_set, test_set


def train_dev_test_split(dataset, dev_fraction, test_fraction, seed=None):
    ''' divide il dataset in train, dev e test set

    Args
    ----
    dataset (Dataset): dataset da dividere
    dev_fraction (float): frazione di utenti da includere nel dev set
    test_fraction: idem
    seed (int): seed da utilizzare nella suddivisione degli utenti

    Ret
    ---
    una tupla con 3 elementi
    training_set (Dataset): il training set
    dev_set: idem
    test_set: idem
    '''
    def divide_user_ids(user_ids):
        dev_n = int(len(user_ids)*dev_fraction)
        test_n = int(len(user_ids)*test_fraction)
        train_n = len(user_ids) - dev_n - test_n

        train_ids = user_ids[:train_n]
        dev_ids = user_ids[train_n: train_n + dev_n]
        test_ids = user_ids[train_n + dev_n:]
        return train_ids, dev_ids, test_ids

    rand_obj = random.Random(seed)

    user_ids = list(dataset.get_user_ids())
    rand_obj.shuffle(user_ids)
    train_user_ids, dev_user_ids, test_user_ids = divide_user_ids(user_ids)
    return train_dev_test_split_by_user(dataset, train_user_ids, dev_user_ids, test_user_ids)


def get_positive_nple_number(dataset, n):
    ''' Calcola il numero di n-ple con ultimo elemento POSITIVO che 
    è possibile comporre utilizzando il dataset

    Args
    ----
    dataset (Dataset): un dataset
    n (int): lunghezza delle tuple da estrarre (compreso il record da classificare)
    '''
    total = 0
    for user_id in dataset.get_user_ids():
        positives, _ = dataset.get_records_with_id(user_id)
        num_positives = len(positives)
        total += binom(num_positives, n-1) * (num_positives-n+1)
    return total


def get_negative_nple_number(dataset, n):
    ''' Calcola il numero di n-ple con ultimo elemento NEGAITIVO che 
    è possibile comporre utilizzando il dataset

    Args
    ----
    dataset (Dataset): un dataset
    n (int): lunghezza delle tuple da estrarre (compreso il record da classificare)
    '''
    total = 0
    for user_id in dataset.get_user_ids():
        positives, negatives = dataset.get_records_with_id(user_id)
        num_positives = len(positives)
        num_negatives = len(negatives)
        total += binom(num_positives, n-1) * num_negatives
    return total


def _get_random_tupla(positives, negatives, n, rand_obj, is_positive):
    ''' Restituisce una lista di n elementi scelti a caso tra i positives e i negatives

    Args
    ----
    positives (Sequence): elementi positivi
    negatives (Sequence): elementi negativi
    n (int): numero di elementi da scegliere
    rand_obj (random.Random): un oggetto di tipo Random
    is_positive (bool): True se anche l'ultimo elemento deve essere positivo

    Ret
    ---
    tupla(list): tupla degli elementi selezionati
    '''
    if is_positive is True:
        tupla = rand_obj.sample(positives, n)
    else:
        # l'ultimo è negativo
        tupla = rand_obj.sample(positives, n-1)
        tupla.append(rand_obj.choice(negatives))

    return tuple(tupla)


def _balance_positives(distinct_tuples, positive_fraction):
    balanced_tuples = []
    wanted_positives = 0
    wanted_negatives = 0
    added_positives = 0
    added_negatives = 0
    total = 0

    # conta positivi e negativi
    for tuple_ in distinct_tuples:
        if tuple_[1] == POSITIVE_RECORD:
            wanted_positives += 1
        else:
            wanted_negatives += 1
        total += 1

    # decide quanti positivi e negativi occorrono
    if (wanted_positives-1) / total > positive_fraction:
        # i positivi sono troppi
        wanted_positives = int(
            positive_fraction*wanted_negatives / (1-positive_fraction))
    elif (wanted_negatives-1) / total > 1 - positive_fraction:
        # i negativi sono troppi
        wanted_negatives = int((1-positive_fraction) *
                               wanted_positives / positive_fraction)

    # compone la lista finale
    for tuple_ in distinct_tuples:
        if tuple_[1] == POSITIVE_RECORD and added_positives < wanted_positives:
            balanced_tuples.append(tuple_)
            added_positives += 1
        elif tuple_[1] == NEGATIVE_RECORD and added_negatives < wanted_negatives:
            balanced_tuples.append(tuple_)
            added_negatives += 1

    return balanced_tuples


def get_tuple_dataset(source_dataset, tuple_len, tuple_num, positive_fraction=0.5, seed=None, info='', max_time_in_minutes=float('inf')):
    ''' Ritorna un dataset di tuple che può essere caricato con in un oggetto TupleDataset

    Args
    ----
    source_dataset (DataSet): il dataset sorgente
    tuple_len (int): lunghezza di ogni tupla (compreso il record da classificare).
                     es. 6 nel caso del 5 + 1
    tuple_num (int): numero di tuple da generare
    positive_fraction (float): frazione di positivi da includere
    seed (int): seme per la generazione del dataset
    info (str): informazioni sul dataset
    max_time_in_minutes (int): termina massimo dopo il numero di minuti specificato

    Ret
    ---
    un dataset di tuple (TupleDataset)
    '''

    def get_indexes(records):
        ''' Ritorna una lista di indici da una lista di record'''
        return [record.index for record in records]

    def get_expected_result(is_positive):
        return POSITIVE_RECORD if is_positive else NEGATIVE_RECORD

    start_time = time.time()
    tuples = []
    user_ids = source_dataset.get_user_ids()
    rand_obj = random.Random(seed)

    # genero le tuple
    for _ in tqdm(range(tuple_num)):
        user_id = rand_obj.choice(user_ids)
        positives, negatives = source_dataset.get_records_with_id(user_id)
        is_positive = rand_obj.choices(
            [True, False], [positive_fraction, 1-positive_fraction])[0]
        tupla = _get_random_tupla(get_indexes(positives), get_indexes(
            negatives), tuple_len, rand_obj, is_positive)
        # è una nuova tupla
        tuples.append((tupla, get_expected_result(is_positive)))

        if (time.time()-start_time) / 60 > max_time_in_minutes:
            break

    distinct_tuples = list(set(tuples))
    distinct_tuples = _balance_positives(distinct_tuples, positive_fraction)
    print(f'numero di tuple generate={len(distinct_tuples)}')

    return TupleDataset.fromTuples(distinct_tuples, source_dataset, info)


class TupleDataset:
    ''' è un dataset di tuple (ad esempio sestuple)'''

    def __init__(self, tuples, source_dataset, info=''):
        ''' Utilizzare un factory method per creare un'istanza di questa classe'''
        self.tuples = tuples
        self.source_dataset = source_dataset
        self.index_to_record_dict = {
            record.index: record for record in source_dataset.records}
        self.info = info

    @classmethod
    def load(cls, path):
        ''' Carica il dataset da un file. Factory method'''
        with open(path, 'rb') as f:
            dict_ = pickle.load(f)
        # ho creato i miei dataset senza salvare il campo info
        # da cui questa 'pezza'
        try:
            info = dict_['info']
        except:
            info = ''
        return cls(dict_['tuples'], dict_['source_dataset'], info)

    @classmethod
    def fromTuples(cls, tuples, source_dataset, info=''):
        ''' Crea un oggetto TupleDataset da una lista di tuple e un
        dataset sorgente. Factory method'''
        return cls(tuples, source_dataset, info)

    def get_field_from_sample(self, index, field):
        ''' Ritorna il sample di indice index
        sostituendo all'intero record il campo field'''
        tupla, expected_result = self[index]
        transformed_tupla = tuple(getattr(record, field) for record in tupla)
        return transformed_tupla, expected_result

    def get_all_fields_from_sample(self, index):
        ''' Ritorna il sample di indice index
        sostituendo all'intero record tutti i campi concatentati'''
        def get_all_fields(record):
            fields = []
            for field in fields:
                fields += getattr(record, field)
            return fields

        tupla, expected_result = self[index]
        transformed_tupla = tuple(get_all_fields(record) for record in tupla)
        return transformed_tupla, expected_result

    def get_custom_sample(self, f, index):
        ''' Ritorna il sample di indice index
        sostituendo a ogni record f(record)'''
        tupla, expected_result = self[index]
        return tuple(f(record) for record in tupla), expected_result

    def save(self, path):
        ''' Salva il file sul path specificato'''
        dict_ = {
            'source_dataset': self.source_dataset,
            'tuples': self.tuples,
            'info': self.info
        }

        with open(path, "wb") as f:
            pickle.dump(dict_, f)

    def __len__(self):
        return len(self.tuples)

    def __getitem__(self, idx):
        ''' Ritorna l'elemento di indice idx del dataset di tuple

        Ret
        ---
        Una tupla con due componenti:
        record_tupla (list): una lista di record
        expected_result (int): 1 se il record da classificare è positivo, altrimenti -1
        '''
        indexes_tupla, expected_result = self.tuples[idx]
        record_tupla = [self.index_to_record_dict[index]
                        for index in indexes_tupla]
        return record_tupla, expected_result

    def shuffle(self):
        ''' Cambia l'ordine delle tuple'''
        random.shuffle(self.tuples)

    def write_info(self):
        ''' Stampa a schermo informazioni sul dataset'''

        num_record = len(self.tuples)
        num_positives = len(
            [tuple_ for tuple_ in self.tuples if tuple_[1] == POSITIVE_RECORD])
        num_negatives = len(
            [tuple_ for tuple_ in self.tuples if tuple_[1] == NEGATIVE_RECORD])

        @dataclass
        class CountUserRecords:
            positives: int
            negatives: int

        info_dict = defaultdict(lambda: CountUserRecords(0, 0))

        for tuple_ in self.tuples:
            record_to_classify_index = tuple_[0][-1]
            record_to_classify = self.index_to_record_dict[record_to_classify_index]
            if record_to_classify.expected_result == POSITIVE_RECORD:
                info_dict[record_to_classify.user_id].positives += 1
            else:
                info_dict[record_to_classify.user_id].negatives += 1

        user_ids = []
        positive_record_nums = []
        negative_record_nums = []

        for user_id in info_dict.keys():
            user_ids.append(user_id)
            positive_record_nums.append(info_dict[user_id].positives)
            negative_record_nums.append(info_dict[user_id].negatives)

        df = pd.DataFrame({
            "user_id": user_ids,
            "positive_record_num": positive_record_nums,
            "negative_record_num": negative_record_nums})

        info = f'''\
            Numero totale record = {num_record}
            Numero record positivi = {num_positives}
            Numero record negativi = {num_negatives}
            info = {self.info}

            '''
        print(textwrap.dedent(info))
        print(df)

#################
# NORMALIZZATORI
#################


def get_field_from_record(self, record, field):
    ''' Ritorna il campo field del record'''
    return getattr(record, field)


def get_all_fields_from_record(record):
    ''' Ritorna tutti i campi concatenati del record'''
    fields = []
    for field in fields:
        fields += getattr(record, field)
    return fields


class StandardNormalizer:
    ''' Normalizza ciascuna feature indipendentemente'''

    def __init__(self, mean_func=np.mean, std_func=np.std):
        ''' è possibile passare funzioni diverse dalla nedia e dalla deviazione standard
        In entrambi i casi devono essere funzioni che prendono in input una lista di numeri e 
        restituiscono un numero. Ad esempio è possibile fare la mediana invece della media '''
        self.mean_func = mean_func
        self.std_func = std_func

    def fit(self, records):
        self.fields_list_dict = defaultdict(list)
        self.means = {}
        self.stds = {}
        for record in records:
            for field in fields:
                self.fields_list_dict[field].append(getattr(record, field))
        for field in fields:
            self.means[field] = [self.mean_func(
                features) for features in zip(*self.fields_list_dict[field])]
            self.stds[field] = [self.std_func(features) for features in zip(
                *self.fields_list_dict[field])]

    def transform(self, record):
        for field in fields:
            features = getattr(record, field)
            for i in range(len(features)):
                features[i] -= self.means[field][i]
                std = self.stds[field][i]
                features[i] /= (std if std != 0 else 1)
        return record


if __name__ == '__main__':
    #################################
    # CREAZIONE DEI DATASET DI TUPLE
    #################################

    dataset_path = r"C:\Users\btlgs\Documents\Project_fraud_2\training_set.csv"
    dataset = DataSet(dataset_path)
    training_set, dev_set, test_set = train_dev_test_split(
        dataset, dev_fraction=0.2, test_fraction=0.2)

    # sestuple
    training_set_classic_sestuple = get_tuple_dataset(
        training_set, tuple_len=6, tuple_num=5_000_000)
    training_set_classic_sestuple.save(
        r"C:\Users\btlgs\Documents\Project_fraud_2\sestuple_datasets\training_set_classic_sestuple")

    dev_set_classic_sestuple = get_tuple_dataset(
        dev_set, tuple_len=6, tuple_num=100_000)
    dev_set_classic_sestuple.save(
        r"C:\Users\btlgs\Documents\Project_fraud_2\sestuple_datasets\dev_set_classic_sestuple")

    test_set_classic_sestuple = get_tuple_dataset(
        test_set, tuple_len=6, tuple_num=100_000)
    test_set_classic_sestuple.save(
        r"C:\Users\btlgs\Documents\Project_fraud_2\sestuple_datasets\test_set_classic_sestuple")

    # coppie
    training_set_classic_coppie = get_tuple_dataset(
        training_set, tuple_len=2, tuple_num=100_000)
    training_set_classic_coppie.save(
        r"C:\Users\btlgs\Documents\Project_fraud_2\coppie_datasets\training_set_classic_coppie")

    dev_set_classic_coppie = get_tuple_dataset(
        dev_set, tuple_len=2, tuple_num=10_000)
    dev_set_classic_coppie.save(
        r"C:\Users\btlgs\Documents\Project_fraud_2\coppie_datasets\dev_set_classic_coppie")

    test_set_classic_coppie = get_tuple_dataset(
        test_set, tuple_len=2, tuple_num=10_000)
    test_set_classic_coppie.save(
        r"C:\Users\btlgs\Documents\Project_fraud_2\coppie_datasets\test_set_classic_coppie")

