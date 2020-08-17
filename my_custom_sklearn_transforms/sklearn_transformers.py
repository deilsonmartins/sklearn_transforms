from sklearn.base import BaseEstimator, TransformerMixin

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        
        # Retornamos um novo dataframe sem as colunas indesejadas
        new_x = data.drop(labels=self.columns, axis='columns')
        
        return new_x

class ADDColumns(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()

        # Criação da nova coluna
        data['H_AULA_PRES/FALTAS'] = round(data['H_AULA_PRES'] / data['FALTAS'], 3)
        
        # Retornamos um novo dataframe 
        return data
 

