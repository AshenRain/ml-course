import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        # if self.oob:
        #     for bag in range(self.num_bags):
        #         indices = np.random.choice(data_length, size=data_length, replace=False)
        #         self.indices_list.append(indices)
        # else:
        for bag in range(self.num_bags):
            indices = np.random.choice(data_length, size=data_length, replace=True)
            self.indices_list.append(indices)
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.

        example:

        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'Все бэги должны быть одинаковой длины!'
        #assert list(map(len, self.indices_list))[0] == len(data), 'Каждый бэг должен содержать `len(data)` элементов!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            indices = self.indices_list[bag]
            data_bag, target_bag = data[indices], target[indices]
            self.models_list.append(model.fit(data_bag, target_bag)) # сохраняем обученные модели здесь
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Получает среднее предсказание для каждого объекта из переданного набора данных
        '''
        predictions = np.zeros(len(data))
        for model in self.models_list:
            predictions += model.predict(data)
        predictions /= len(self.models_list)
        return predictions
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        # Your Code Here

        for bag in range(self.num_bags):
            indices = self.indices_list[bag]
            mask = np.ones(len(self.data), dtype=bool)
            mask[indices] = False
            oob_data = self.data[mask]
            model = self.models_list[bag]
            if len(oob_data) > 0:
                predictions = model.predict(oob_data)
                for i, index in enumerate(np.where(mask)[0]):
                    list_of_predictions_lists[index].append(predictions[i])

        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        '''
        Вычисляет среднее предсказание для каждого объекта из тренировочного набора.
        Если объект использовался во всех бэгах во время обучения, возвращается None вместо предсказания.
        '''
        self._get_oob_predictions_from_every_model()
        n_models_used_for_oob = np.sum(self.list_of_predictions_lists != [])
        self.oob_predictions = np.zeros(len(self.data))
        for i in range(len(self.data)):
            if len(self.list_of_predictions_lists[i]) > 0:
                self.oob_predictions[i] = np.mean(self.list_of_predictions_lists[i])
            else:
                self.oob_predictions[i] = None
        
    def OOB_score(self):
        '''
        Вычисляет среднеквадратичную ошибку для всех объектов, у которых есть хотя бы одно предсказание
        '''
        self._get_averaged_oob_predictions()
        return np.nanmean((self.oob_predictions - self.target) ** 2)
