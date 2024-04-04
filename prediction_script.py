def predict(audio_file_path):
    import numpy as np
    from tensorflow.keras.models import Sequential, model_from_json
    json_file = open('CNN_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("best_model1_weights.h5")
    print("Loaded model from disk")
    # predict()
    import pickle

    with open('scaler2.pickle', 'rb') as f:
        scaler2 = pickle.load(f)

    with open('encoder2.pickle', 'rb') as f:
        encoder2 = pickle.load(f)


    print("Done")

    import librosa
    def zcr(data,frame_length,hop_length):
        zcr=librosa.feature.zero_crossing_rate(data,frame_length=frame_length,hop_length=hop_length)
        return np.squeeze(zcr)
    def rmse(data,frame_length=2048,hop_length=512):
        rmse=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
        return np.squeeze(rmse)
    def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
        mfcc=librosa.feature.mfcc(y=data,sr=sr)
        return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

    def extract_features(data,sr=22050,frame_length=2048,hop_length=512):
        result=np.array([])

        result=np.hstack((result,
                        zcr(data,frame_length,hop_length),
                        rmse(data,frame_length,hop_length),
                        mfcc(data,sr,frame_length,hop_length)
                        ))
        return result

    def get_predict_feat(path):
        d, s_rate= librosa.load(path, duration=2.5, offset=0.6)
        res=extract_features(d)
        result=np.array(res)
        print(result.size)
        while(result.size < 2376):
            result = np.append(result, 0)
        print(result.size)
        result=np.reshape(result,newshape=(1,2376))
        i_result = scaler2.transform(result)
        final_result=np.expand_dims(i_result, axis=2)

        return final_result

    # res=get_predict_feat("algorithm/Datasets/Audio_Speech_Actors_01-24/Actor_02/03-01-01-01-01-01-02.wav")
    # print(res.shape)
    emotions1={1:'Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Angry', 6:'Fear', 7:'Disgust',8:'Surprise'}
    def prediction(path1):
        res=get_predict_feat(path1)
        predictions=loaded_model.predict(res)
        y_pred = encoder2.inverse_transform(predictions)
        return y_pred[0][0]
    return prediction(audio_file_path)
