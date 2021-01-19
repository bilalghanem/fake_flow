The is the code of the system proposed in the paper:

# # FakeFlow: Fake News Detection by Modeling the Flow of Affective Information



REQUIREMENTS:
- gensim==3.8.0
- joblib==0.14.1
- Keras==2.2.4
- Keras-Preprocessing==1.1.0
- keras-self-attention==0.35.0
- numpy==1.16.0
- pandas==0.24.2
- nltk==3.4.5
- scikit-learn==0.20.2
- tensorflow-gpu==1.14.0
- tqdm==4.32.1
- hyperopt==0.1.1





Place your data in the folder `./data/DATASET_NAME`

To run the model, run the file: `fake_flow.py`

parameters:
`-d`: dataset name (i.e. MultiSourceFake).

`-sn`: number of segments.

`-s`: to search for params; enter a number larger than 0 to search for N different combination of parameters (e.g. 150).

`-m`: mode (train or test); if you want to load a pretrained model.

An example:
> fake_flow.py -d MultiSourceFake -sn 10

To load saved model after training:
> fake_flow.py -d MultiSourceFake -sn 10 -m test

To search for best params:
> fake_flow.py -d MultiSourceFake -s 80



Citation:

    @inproceedings{ghanem2021fakeflow,
      title={{FakeFlow: Fake News Detection by Modeling the Flow of Affective Information}},
      author={Ghanem, Bilal and Ponzetto, Simone Paolo and Rosso, Paolo and Rangel, Francisco},
      booktitle={Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics},
      year={2021}
    }
