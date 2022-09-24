import pandas as pd

def read_from_csv(path, dataset):
    
    proc = {"BIDMC":process_BIDMC}
    # print(dataset)
    
    return proc[dataset](path)
    # ppg,ecg = func()
    
    # return process_BIDMC(path)
    

def process_BIDMC(path):
    df = pd.read_csv(path)
    columns = [' RESP', ' V', ' AVR']
    # df.drop(columns, inplace=True, axis=1)
    t_ref = df['Time [s]'].to_numpy()
    ppg = df[' PLETH'].to_numpy()
    ecg = df[' II'].to_numpy()
    
    return ppg,ecg
    

    
    