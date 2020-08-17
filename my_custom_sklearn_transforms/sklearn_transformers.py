from sklearn.base import BaseEstimator, TransformerMixin

from imblearn.over_sampling import SMOTE

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        x, y = X
        data = x.copy()
        
        # Retornamos um novo dataframe sem as colunas indesejadas
        new_x = data.drop(labels=self.columns, axis='columns')
        
        return new_x, y

class ADDColumns(BaseEstimator, TransformerMixin):

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()

        # Criação da nova coluna
        data['H_AULA_PRES/FALTAS'] = round(data['H_AULA_PRES'] / data['FALTAS'], 3)
        
        # Retornamos um novo dataframe 
        return data, y
 
class Balanceamento(BaseEstimator, TransformerMixin):
  
    def fit(self, X):
        return self
    
    def transform(self, X):
       
        x, y = X
        # Instanciando o SMOTE
        nr = SMOTE()
        
        # Pegando os dados balanceados
        new_X, new_y = nr.fit_sample(x, y)

        # Retornamos um novo dataframe
        return new_X, new_y
