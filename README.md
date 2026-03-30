# MachineLearning_VT2026
Final Assignment for my Machine Learning course

Discussion:
In my first runs with a vector dimension of 100 and 10 epochs for the FastText model as well as 10 epochs for the neural-network model, the accuracy for the test set amounted to 36.27%. When I increased the dimension to 200 and the epochs for both models to 30, the accuracy for the test set was 66.18%. With a dimension of 300 and 50 epochs, the accuracy for the test set was 70.10%. Therefore it seems that increasing both the dimensionality of the vector embeddings as well as the number of epochs increases the representation quality of the vectors and therefore helps the neural-network model learn more and yields significantly better results.

For the discussion of the confusion matrix, I will focus on the one from the run with the best result (70.10% accuracy): The categories "science/technology" and "travel" were predicted correctly most often. But there is a strong diagonal visible, which shows the correct predictions. This means that every category has been predicted correctly at least a few times (10 times in this case).
It is notable that in the run with 10 epochs, the model predicted only 4 out of the 7 categories with a focus on 3 of them ("travel", "science/technology", "politics"), according to the confusion matrix. Therefore the improvement when using 50 epochs is very clear, since all 7 classes were actually predicted in that run.
There are still some misclassifications like "health" and "entertainment" being predicted as "science/technology", but overall the predictions are relatively balanced.

The model generally performed better than chance (14.29%) even on my worst run (36.27% accuracy). With the best accuary of 70.10% it actually performed significantly better than chance, which shows that it actually learned differences between the categories. However, higher dimensional vector embeddings and longer training were essential in that process.

Bonus Part 1:
The graphs tracking the performance during the training loop show that the training loss decreases steadily, after a sharp drop ca. after the first 5 epochs. The accuracy increases stabilizes around 65-70% (with small fluctuations, but generally an upwards trend).



Instructions on how to run my scripts:

Script 1: train_fasttext.py
- takes any number of tsv files (but at least one)
- user has to specify vector size (embedding dimension)
- takes an output path to later save the trained FastText model
- optionally change more parameters for the FastText model
example (optional arguments in parentheses):

python train_fasttext.py --input_files train.tsv dev.tsv test.tsv --dim 200 --output_path fasttext.model (--win 5 --min_count 1 --epochs 30)

Script 2: sentence_embeddings.py
- takes an input tsv file (one at a time)
- takes the model output path from the previously trained FastText model
- takes output file name to later save the vector embeddings as
example:

python sentence_embeddings.py --input_file train.tsv --output_path fasttext.model --output_file train_embeddings.npy

python sentence_embeddings.py --input_file test.tsv --output_path fasttext.model --output_file test_embeddings.npy

python sentence_embeddings.py --input_file dev.tsv --output_path fasttext.model --output_file dev_embeddings.npy

Script 3: topic_classifier.py
- takes the number of epochs
- takes the batch size
- takes the output path in order to later save the trained model weights
- optionally change the default input and embeddings files
- optionally use the validation set to evaluate the model's perfomance during the training loop
example (optional arguments in parentheses):

python topic_classifier.py --epochs 30 --batch_size 4 --output_path classifier.model (--input_file train.tsv --embeddings_file train_embeddings.npy --dev_file dev.tsv --dev_embeddings dev_embeddings.npy)

Script 4: evaluation.py
- takes the model path from the previously trained neural-network model weights
- takes batch size
- optionally change default input and embeddings files
example (optional arguments in parentheses):

python evaluation.py --model_path classifier.model --batch_size 4 (--input_file test.tsv --embeddings_file test_embeddings.npy)


Transcript of one full run:

[gusbeihel@GU.GU.SE@mltgpu-2 ~]$ python train_fasttext.py --input_files train.tsv dev.tsv test.tsv --dim 300 --output_path fasttext.model --win 5 --min_
count 1 --epochs 50
* Loading data *
* Training FastText *
* Saving model *
All done!
[gusbeihel@GU.GU.SE@mltgpu-2 ~]$ python sentence_embeddings.py --input_file train.tsv --output_path fasttext.model --output_file train_embeddings.npy
* Loading model *
* Reading data *
* Generating sentence embeddings *
* Saving vector embeddings to train_embeddings.npy *
All done!
[gusbeihel@GU.GU.SE@mltgpu-2 ~]$ python sentence_embeddings.py --input_file test.tsv --output_path fasttext.model --output_file test_embeddings.npy
* Loading model *
* Reading data *
* Generating sentence embeddings *
* Saving vector embeddings to test_embeddings.npy *
All done!
[gusbeihel@GU.GU.SE@mltgpu-2 ~]$ python sentence_embeddings.py --input_file dev.tsv --output_path fasttext.model --output_file dev_embeddings.npy
* Loading model *
* Reading data *
* Generating sentence embeddings *
* Saving vector embeddings to dev_embeddings.npy *
All done!
[gusbeihel@GU.GU.SE@mltgpu-2 ~]$ python topic_classifier.py --epochs 50 --batch_size 4 --output_path classifier.model --dev_file dev.tsv --dev_embeddings de
v_embeddings.npy
* Loading sentence embeddings *
* Encoding labels *
* Creating dataset *
* Creating dataloader *
* Building model *
* Adding optimizer *
* Training data *
Epoch: 1
Loss (average): 1.5987041965126991

* Computing accuracy *
Accuracy: 0.5454545454545454

Epoch: 2
Loss (average): 1.1565237195993012

* Computing accuracy *
Accuracy: 0.6161616161616161

Epoch: 3
Loss (average): 0.9780269947156988

* Computing accuracy *
Accuracy: 0.6161616161616161

Epoch: 4
Loss (average): 0.8927171552045778

* Computing accuracy *
Accuracy: 0.6161616161616161

Epoch: 5
Loss (average): 0.8422452069141648

* Computing accuracy *
Accuracy: 0.6363636363636364

Epoch: 6
Loss (average): 0.7884927360679616

* Computing accuracy *
Accuracy: 0.6060606060606061

Epoch: 7
Loss (average): 0.7665952105413784

* Computing accuracy *
Accuracy: 0.6464646464646465

Epoch: 8
Loss (average): 0.7304181429049508

* Computing accuracy *
Accuracy: 0.6464646464646465

Epoch: 9
Loss (average): 0.7024892720563168

* Computing accuracy *
Accuracy: 0.6161616161616161

Epoch: 10
Loss (average): 0.6905403150330213

* Computing accuracy *
Accuracy: 0.6464646464646465

Epoch: 11
Loss (average): 0.6580962938129563

* Computing accuracy *
Accuracy: 0.6464646464646465

Epoch: 12
Loss (average): 0.6406339267057113

* Computing accuracy *
Accuracy: 0.6767676767676768

Epoch: 13
Loss (average): 0.6318800051019273

* Computing accuracy *
Accuracy: 0.6868686868686869

Epoch: 14
Loss (average): 0.6118843268433755

* Computing accuracy *
Accuracy: 0.6464646464646465

Epoch: 15
Loss (average): 0.5910286061381075

* Computing accuracy *
Accuracy: 0.6666666666666666

Epoch: 16
Loss (average): 0.5846620375629176

* Computing accuracy *
Accuracy: 0.6666666666666666

Epoch: 17
Loss (average): 0.5577437428291887

* Computing accuracy *
Accuracy: 0.7070707070707071

Epoch: 18
Loss (average): 0.5483048955581828

* Computing accuracy *
Accuracy: 0.6767676767676768

Epoch: 19
Loss (average): 0.5309048378522593

* Computing accuracy *
Accuracy: 0.696969696969697

Epoch: 20
Loss (average): 0.5035308564970777

* Computing accuracy *
Accuracy: 0.696969696969697

Epoch: 21
Loss (average): 0.5012653030624444

* Computing accuracy *
Accuracy: 0.6767676767676768

Epoch: 22
Loss (average): 0.49353250630453904

* Computing accuracy *
Accuracy: 0.6767676767676768

Epoch: 23
Loss (average): 0.4727708787911318

* Computing accuracy *
Accuracy: 0.6868686868686869

Epoch: 24
Loss (average): 0.46358483751431445

* Computing accuracy *
Accuracy: 0.6464646464646465

Epoch: 25
Loss (average): 0.438597224668642

* Computing accuracy *
Accuracy: 0.6565656565656566

Epoch: 26
Loss (average): 0.42983230226673186

* Computing accuracy *
Accuracy: 0.696969696969697

Epoch: 27
Loss (average): 0.42050802093581297

* Computing accuracy *
Accuracy: 0.6868686868686869

Epoch: 28
Loss (average): 0.4020718245618892

* Computing accuracy *
Accuracy: 0.6363636363636364

Epoch: 29
Loss (average): 0.3890565652611919

* Computing accuracy *
Accuracy: 0.6767676767676768

Epoch: 30
Loss (average): 0.3806235458283812

* Computing accuracy *
Accuracy: 0.6868686868686869

Epoch: 31
Loss (average): 0.36540449559900234

* Computing accuracy *
Accuracy: 0.6565656565656566

Epoch: 32
Loss (average): 0.35342745160133665

* Computing accuracy *
Accuracy: 0.6565656565656566

Epoch: 33
Loss (average): 0.348125857734968

* Computing accuracy *
Accuracy: 0.6666666666666666

Epoch: 34
Loss (average): 0.32785024208186025

* Computing accuracy *
Accuracy: 0.6565656565656566

Epoch: 35
Loss (average): 0.33472537953903986

* Computing accuracy *
Accuracy: 0.6565656565656566

Epoch: 36
Loss (average): 0.30116851832479535

* Computing accuracy *
Accuracy: 0.6565656565656566

Epoch: 37
Loss (average): 0.29421381319116335

* Computing accuracy *
Accuracy: 0.6464646464646465

Epoch: 38
Loss (average): 0.27947418213575886

* Computing accuracy *
Accuracy: 0.6666666666666666

Epoch: 39
Loss (average): 0.26920147469817574

* Computing accuracy *
Accuracy: 0.6767676767676768

Epoch: 40
Loss (average): 0.26646017521588045

* Computing accuracy *
Accuracy: 0.6666666666666666

Epoch: 41
Loss (average): 0.2524982827281664

* Computing accuracy *
Accuracy: 0.6666666666666666

Epoch: 42
Loss (average): 0.23957142083831554

* Computing accuracy *
Accuracy: 0.6565656565656566

Epoch: 43
Loss (average): 0.23236126361271917

* Computing accuracy *
Accuracy: 0.6666666666666666

Epoch: 44
Loss (average): 0.21960191812435037

* Computing accuracy *
Accuracy: 0.6363636363636364

Epoch: 45
Loss (average): 0.22096484765643254

* Computing accuracy *
Accuracy: 0.6565656565656566

Epoch: 46
Loss (average): 0.203841852233216

* Computing accuracy *
Accuracy: 0.6767676767676768

Epoch: 47
Loss (average): 0.19820376109088433

* Computing accuracy *
Accuracy: 0.6565656565656566

Epoch: 48
Loss (average): 0.17984858793484323

* Computing accuracy *
Accuracy: 0.6464646464646465

Epoch: 49
Loss (average): 0.17117366958204497

* Computing accuracy *
Accuracy: 0.6666666666666666

Epoch: 50
Loss (average): 0.16810000272446565

* Computing accuracy *
Accuracy: 0.6464646464646465

* Evaluating performance *
All done!
[gusbeihel@GU.GU.SE@mltgpu-2 ~]$ python evaluation.py --model_path classifier.model --batch_size 4
* Loading sentence embeddings *
* Encoding labels *
* Creating dataset *
* Creating dataloader *
* Loading trained weights *
/home/gusbeihel@GU.GU.SE/evaluation.py:103: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(args.model_path))
* Predicting *
* Computing accuracy *
Accuracy: 0.7402
All done!
[gusbeihel@GU.GU.SE@mltgpu-2 ~]$

NB: This trial run on the server yielded an even higher accuracy of 74.02% for the test set.


The scripts have been tested on mltgpu-2 and adjusted accordingly. Both the confusion matrix and the plot for Bonus Part 1 are saved as png files, since they do not show on the server.
