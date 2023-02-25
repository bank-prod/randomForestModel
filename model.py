import pandas as pd
class Model :
    def __init__(self,encoders,model) :
        self.encoders = encoders
        self.model = model
        
    def __validate_data(self,X) :
        return True
    
    def preprocessing(self,X) : 
        df = pd.DataFrame.from_dict([X])
        
        df['Gender'] = self.encoders['gender'].transform(df['Gender'])
        
        geo = self.encoders['geo'].transform(df['Geography'])
        df = df.drop(['Geography'], axis = 1)
        
        column_geography = ['Geography_'+ val for val in self.encoders['geo'].classes_]
        df_geography = pd.DataFrame(geo,columns=column_geography, index=df.index)
        df = df.join(df_geography)

        return df
    
    def predict(self,X) :
        if not self.__validate_data(X):
            raise Exception('Données non valides')
        X_enc = self.preprocessing(X)
        prob = self.model.predict_proba(X_enc)
        pred = self.model.predict(X_enc)[0]
        label_ind,proba = prob.argmax(), prob.max()
        classe = self.encoders['exited'].inverse_transform([pred])[-1] 
            
        return classe, proba
    