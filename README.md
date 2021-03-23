# Novel-Text-Generation-with-LSTM

For this project, entire books are used as inputs in a Recurrent Neural Network with Long-Short Term Memory layers. The output will be text generation that matches the tone and vibe of the novels/plays.


## Dataset and Library
PyTorch was the library used for this project.
The books used for this Text Generator were: multiple Shakespear plays, War & Peace, The Adventures of Tom Sawyer, Pride & Prejudice, and Meditations. 

## Encoding the Data
Networks cannot understand raw text, so upon reading in the text data, all the characters must be encoded, such as [A B C D] = [0 1 2 3]. This means every unique character will be assigned an encoded value.
As such, a one-hot-encoder function was created to return the encoded text data as a numpy array of size: number of text by number of unique character. In this array, each row corresponds to each the new text, and each column will be 0 or 1 depending on the unique character that was found in this text slot. For more details on one hot encoding, please see this example on Stack Overflow:
https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array

## Generate Batches Function
For the one-hot-encoder function, text will be fed in as batches rather than the entire text at once. The batches will be based on a sequence of characters, shifted over by one step. This was found to be more effective than having the label, y, just be the final character. This is because the actual grammatical structure of the characters is considered and don’t just end up predicting the most common letter every time. This allows the network to learn long term structure, not just in the training features, but also in the label that it’s going to be producing.
The generate-batches function takes the samples per batch, sample sequence length, and the encoded texts as arguments. The function uses these arguments to perform array slicing to output the batches as x and labels as y.

## The RNN Model
The RNN model by default takes 4 LSTM layers with 256 as the hidden layer size. The LSTM layers are followed up by a dropout layer, which output to a fully connected layer. The final output of the fully connected layer is the generated text.
Adam was used as the optimizer. The loss measurement was Cross Entropy Loss, since this is a categorical problem. 

## Results
Please visit the following link to for more results: https://hjmok.github.io/josephmok_portfolio/#/TG 
