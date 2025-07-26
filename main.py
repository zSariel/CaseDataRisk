import pandas as pd 
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

modelo = RandomForestClassifier()

#usando pandas para analisar as pastas em csv, usar o numpy para ler a quantidade de dados fornecidos, estou usando o scikit-learn para conseguir fazer a previsão, para resolver os problemas de classificação.#

#usando data frame para ler na mesma pasta que o programa main.py#

base_cadastral = pd.read_csv ("./base_cadastral.csv" , sep=";")
base_info = pd.read_csv ("./base_info.csv" , sep=";")
pagamentos_desenvolvimento = pd.read_csv ("./base_pagamentos_desenvolvimento.csv" , sep=";")
pagamentos_teste = pd.read_csv ("./base_pagamentos_teste.csv" , sep=";")

#usando do sep= ";" para reconhecer os sepadores nas planilhas, que estava com esse problema"

def padronizar_colunas(df):
    df.columns = [col.strip().upper() for col in df.columns]
    return df

base_cadastral = padronizar_colunas(base_cadastral)
base_info = padronizar_colunas(base_info)
base_pagamentos_desenvolvimento = padronizar_colunas(pagamentos_desenvolvimento)
base_pagamentos_teste = padronizar_colunas(pagamentos_teste)

#usando o def padronizar_colunas, para conseguir padronizar e organizar as pasta caso der algum erro, estava dando erro e foi a solução que consegui achar para arrumar isso, e assim fica mais facil caso for adicionar mais pastas e conseguir alterar algumas partes.#

pagamentos_desenvolvimento['DATA_VENCIMENTO'] = pd.to_datetime(pagamentos_desenvolvimento["DATA_VENCIMENTO"] , errors='coerce')
pagamentos_desenvolvimento["DATA_PAGAMENTO"] = pd.to_datetime(pagamentos_desenvolvimento["DATA_PAGAMENTO"] , errors='coerce')
#Estou usando o erros='coerce' por não ter certeza que os dados estão perfeitos, e isso iria impedir que o codigo tivesse alguma falha#

pagamentos_desenvolvimento ['DIAS_ATRASO'] =(pagamentos_desenvolvimento['DATA_PAGAMENTO']- pagamentos_desenvolvimento["DATA_VENCIMENTO"]).dt.days
pagamentos_desenvolvimento["INADIMPLENTE"] = (pagamentos_desenvolvimento["DIAS_ATRASO"] >=5).astype(int)

#usei o dt.days para extrair o numero de dias, então ele calcula que se atraso for maior que >=5, vai declarar que está inadimplente, caso não ocorra ele retorna como adimplente.
#print(pagamentos_desenvolvimento[['DATA_PAGAMENTO', 'DATA_VENCIMENTO', 'DIAS_ATRASO', 'INADIMPLENTE']].head(10)) usei esse comando para testar até agora o codigo#

def preprocess(df):
    df = df.copy() #prerocess para conseguir novas colunas, usando o df.copy para evitar modificar dataframe que ja foi usado usando.#
    if "DATA_EMISSAO_DOCUMENTO" in df.columns:
        df["DATA_EMISSAO_DOCUMENTO"] = pd.to_datetime(df["DATA_EMISSAO_DOCUMENTO"], errors='coerce')
        #uso essa parte para calcular as datas e se não existir, vai apenas ignorar #
    if "DATA_VENCIMENTO" in df.columns:
        df["DATA_VENCIMENTO"] = pd.to_datetime(df["DATA_VENCIMENTO"], errors='coerce')
        df["DIA_VENCIMENTO"] = df["DATA_VENCIMENTO"].dt.day
        #Isso vai ajudar a entender melhor se o cliente costuma atrasar mais os vencimentos no mes#
    else:
        df["DIA_VENCIMENTO"] = 0
    return df

pag_desenvolvimento = preprocess(pagamentos_desenvolvimento)
pagamentos_teste = preprocess(pagamentos_teste)

base_total = pag_desenvolvimento.merge(base_info , on = ["ID_CLIENTE", "SAFRA_REF"], how='left')
base_total = base_total.merge(base_cadastral, on ="ID_CLIENTE", how = 'left') 

#nessa parte to juntando os dados para fazer uma unica base de dados para conseguir criar a submissao, uso o tipo de junçãoo left e uso a segunda linha para não ficar repetindo os mesmo clientes que varia por mes, tendo uma analise por cliente#

#aqui ja ta criando o arquiv com as colunas certa e rodando por enquanto, agora adicionar alguns dados para conseguir fazer previsão de dados, adicionando a biblioteca Scikit-learn#

cat_cols = ["DOMINIO_EMAIL", "SEGMENTO_INDUSTRIAL", "PORTE"]
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    base_total[col] = le.fit_transform(base_total[col].astype(str))
    le_dict[col] = le
#transforma as variáveis categorias em números inteiros, para que RandomForestClassifier conseguia rodar#

features = [
    "VALOR_A_PAGAR", "TAXA", "RENDA_MES_ANTERIOR", "NO_FUNCIONARIOS",
    "DIA_VENCIMENTO", "DOMINIO_EMAIL", "SEGMENTO_INDUSTRIAL", "PORTE"
]

X = base_total[features].fillna(0)
y = base_total["INADIMPLENTE"]
#aqui ta definindo as colunas inadimplente e adimplente, sendo x a adimplente e y a inadimplente

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) 
#devide os dados em 80% para treino e os 20% para validação

model = RandomForestClassifier(n_estimators=100, random_state=42) 
model.fit(X_train, y_train)
# usei 100 árvores pois é um valor comum que tende a dar bons resultados sem custar muito desempenho, tentei usar também com 50 e 200 árvores porem com 50 o modelo teve desempenho inferior (ROC AUC mais baixo), e com 200 aumentou muito o tempo de processamento sem ganho relevante.

y_pred_proba = model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_pred_proba)
print(f"ROC AUC na validação: {roc_auc:.4f}") 
#gera as probabilidades de inadimplência e calcula a métrica roc auc

teste = pagamentos_teste.merge(base_info, on=["ID_CLIENTE", "SAFRA_REF"], how="left")
teste = teste.merge(base_cadastral, on="ID_CLIENTE", how="left")
teste = preprocess(teste)
#junta os dados da base teste com as outras informações

teste[cat_cols] = teste[cat_cols].astype(str)
for col in cat_cols:
    teste[col] = le_dict[col].transform(teste[col])

X_test = teste[features].fillna(0)
teste["PROBABILIDADE_INADIMPLENCIA"] = model.predict_proba(X_test)[:, 1]
#aplica os mesmo rótulos do treino à base, garantindo que seja formatada igual
submissao = (
    teste[["ID_CLIENTE", "SAFRA_REF", "PROBABILIDADE_INADIMPLENCIA"]]
    .groupby(["ID_CLIENTE", "SAFRA_REF"], as_index=False)
    .mean()
)
submissao.to_csv("submissao_case.csv", index=False)
print("Finalizado o arquivo 'submissao_case.csv' foi um sucesso.")