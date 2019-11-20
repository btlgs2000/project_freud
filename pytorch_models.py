import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.set_printoptions(profile="full")

##################################################
# Modelli con rete comune più layer di averaging
##################################################


class AveragingLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs):
        ''' Esegue una media sui tensori
        Args
        ----
        xs (list) : lista di tensori
        '''
        return torch.mean(torch.stack(xs), dim=0)


class MlpNet(nn.Module):
    ''' Generica rete MLP

    Attrs
    -----
    layers_n (List[int]): lista con il numero di neuroni per ciascuno strato
    activation (Callable): funzione di attivazione per gli strati interni
    last_activation (Callable): funzione di attivazione per l'ultimo strato
    '''

    def __init__(self, layers_n, activation, last_activation):
        super().__init__()
        self.layers_n = layers_n
        self.activation = activation
        self.last_activation = last_activation
        for i, (in_features, out_features) in enumerate(zip(layers_n[:-1], layers_n[1:])):
            linear_layer = nn.Linear(in_features, out_features)
            self.add_module(f"linear layer {i}", linear_layer)

    def forward(self, x):
        list_of_layers = list(self.children())
        for linear_layer in list_of_layers[:-1]:
            x = self.activation(linear_layer(x))
        last_layer = list_of_layers[-1]
        x = self.last_activation(last_layer(x))
        return x


class SplitterLayer(nn.Module):
    '''Prende in input un tensore bidimensionale,
    in cui la prima dimensione è quella del batch e lo splitta nella seconda dimensione

    Attrs
    -----
    arrays_lengths (List[int]): ogni intero rappresenta la dimensione di un array
    '''

    def __init__(self, arrays_lengths):
        super().__init__()
        self.arrays_lengths = arrays_lengths

    def forward(self, x):
        ouputs = []
        cumulative_lengths = [0] + np.cumsum(self.arrays_lengths).tolist()
        for i in range(len(cumulative_lengths)-1):
            output = x[:, cumulative_lengths[i]:cumulative_lengths[i+1]]
            ouputs.append(output)
        return ouputs


class SplitterModel(nn.Module):
    ''' Splitta ciascun record in n parti e fa elabrare ciascuna
    parte ad un diverso modello. Infine utilizza come classificatore uno strato lineare '''

    def __init__(self, arrays_lengths, models):
        super().__init__()
        # lunghezze degli array
        self.arrays_lengths = arrays_lengths
        # prende una batch e restituisce 6 batch (uno per ogni array)
        self.splitter_layer = SplitterLayer(arrays_lengths)
        self.models = models
        for i, parallel_model in enumerate(models):
            self.add_module(f"parallel_model {i}", parallel_model)
        self.linear = nn.Linear(len(arrays_lengths), 1)

    def forward(self, example_records, record_to_classify):
        n = example_records.size()[1]  # nunmero di record di esempio
        example_splitted = [self.splitter_layer(
            example_records[:, i, :]) for i in range(n)]
        one_splitted = self.splitter_layer(record_to_classify)
        outputs = []
        for i in range(len(self.arrays_lengths)):
            # per ogni array
            one = one_splitted[i]
            five = torch.stack([example_splitted[j][i]
                                for j in range(5)], dim=1)
            output = self.models[i](five, one)
            outputs.append(output)
        return torch.sigmoid(self.linear(torch.stack(outputs, dim=1))).view(-1)


class ParallelModel(nn.Module):
    """ Modello che esegue la stessa rete sui record di esempio e sul record da classificare.
    Poi esegue il Layer di averaging sull'output dei record di esempio e ritorna l'output sull'averaging
    e sul record da classificare"""

    def __init__(self, sub_net, averaging_layer, classifier_layer):
        '''
        sub_net (nn.Module) : rete da applicare a tutti i record (n + 1)
        averaging_layer (nn.Module) : 
        classifier_layer (nn.Module) : modulo che ritorna la probabilità che il record sia positivo (non fraudolento)'''
        super().__init__()
        self.sub_net = sub_net
        self.averaging_layer = averaging_layer
        self.classifier_layer = classifier_layer

    def forward(self, example_records, record_to_classify):
        '''
        example_records : tensore PyTorch di dimensioni batch x n (num di record) x num_fields (caso standard)
        record_to_classify : dimensioni batch x num_fields
        label : 0 se fraudolento, 1 se dell'utente
        '''
        n = example_records.size()[1]  # nunmero di record di esempio
        n_outputs = [self.sub_net(example_records[:, i, :]) for i in range(n)]
        one_outputs = self.sub_net(record_to_classify)
        average_outputs = self.averaging_layer(n_outputs)
        return self.classifier_layer(torch.cat((average_outputs, one_outputs), dim=1)).view(-1)


class SingleFeatureModel(nn.Module):
    ''' Esegue una classificazione su una singola feature'''

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, example_records, record_to_classify):
        x = torch.cat((example_records, record_to_classify.view(-1, 1)), dim=1)
        return self.module(x).view(-1)


class PerFeatureModel(nn.Module):
    ''' Esegue lo stesso modello singleFeatureModel per ogni feature
    e alla fine dà in input i risultati ad un classificatore'''

    def __init__(self, singleFeatureModel, classifier):
        super().__init__()
        self.classifier = classifier
        self.singleFeatureModel = singleFeatureModel
        # for param in self.singleFeatureModel.parameters():
        #     param.requires_grad = False

    def forward(self, example_records, record_to_classify):
        record_to_classify = torch.unsqueeze(record_to_classify, dim=1)
        outs = []
        dims = example_records.size()[2]
        for i in range(dims):
            outs.append(self.singleFeatureModel(
                example_records[:, :, i], record_to_classify[:, :, i]))
        classifier_input = torch.stack(outs, dim=1)
        return self.classifier(classifier_input).view(-1)


class PerFeatureModelMulti(nn.Module):
    ''' Come PerFeatureModel ma le diverse feature vengono elaborate 
    con reti differenti (una rete per ogni feature)'''

    def __init__(self, singleFeatureModels, classifier):
        super().__init__()
        self.classifier = classifier
        self.singleFeatureModels = singleFeatureModels
        for i, model in enumerate(singleFeatureModels):
            self.add_module(f"model {i}", model)
        # for param in self.singleFeatureModel.parameters():
        #     param.requires_grad = False

    def forward(self, example_records, record_to_classify):
        record_to_classify = torch.unsqueeze(record_to_classify, dim=1)
        outs = []
        dims = example_records.size()[2]
        for i in range(dims):
            outs.append(self.singleFeatureModels[i](
                example_records[:, :, i], record_to_classify[:, :, i]))
        classifier_input = torch.stack(outs, dim=1)
        # print(classifier_input)
        return self.classifier(classifier_input).view(-1)


class PerFeatureModelDiscretize(nn.Module):
    ''' Come PerFeatureModel ma gli output del singleFeatureModel vengono
    discretizzati prima di essere trasmessi al classificatore'''

    def __init__(self, singleFeatureModel, classifier):
        super().__init__()
        self.classifier = classifier
        self.singleFeatureModel = singleFeatureModel
        for param in self.singleFeatureModel.parameters():
            param.requires_grad = False

    def forward(self, five_records, record_to_classify):
        record_to_classify = torch.unsqueeze(record_to_classify, dim=1)
        outs = []
        dims = five_records.size()[2]
        for i in range(dims):
            outs.append(self.singleFeatureModel(
                five_records[:, :, i], record_to_classify[:, :, i]))
        classifier_input = torch.stack(outs, dim=1)
        # print(classifier_input)
        return self.classifier(torch.round(classifier_input) - 0.5).view(-1)
