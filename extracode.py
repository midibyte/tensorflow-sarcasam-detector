comment_model = keras.Sequential()
	comment_model.add(hub_layer)
	comment_model.add(tf.keras.layers.Dense( 16, activation='relu' ))
	comment_model.add(tf.keras.layers.Dense( 4, activation='relu' ))


	subreddit_model = keras.Sequential()
	subreddit_model.add(hub_layer_subreddit)
	subreddit_model.add(tf.keras.layers.Dense( 16, activation='relu' ))
	subreddit_model.add(tf.keras.layers.Dense( 4, activation='relu' ))

	add_models = keras.layers.add( [ comment_model.output, subreddit_model.output ] )
	

	#merge models
	result_model = keras.Sequential()
	# merged_models = tf.keras.Sequential()
	# models_to_merge = list()
	# models_to_merge.append(subreddit_model)
	# models_to_merge.append(comment_model)
	# merged_models.add( tf.keras.layers.concatenate( models_to_merge, axis=-1) )

	# result_model.add(merged_models)
	result_model.add( tf.keras.layers.Dense(1, activation='sigmoid') )
	
	result_out = result_model(add_models)

	final_model = keras.models.Model( [ comment_model.input, subreddit_model.input ], result_out )

	final_model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
	model = final_model
	# model.fit( [] )


	# out = keras.Dense( shape=(1, ), activation='sigmoid' )

	# setting up keras model
	# model = tf.keras.Sequential()
	# # model = keras.Model(inputs=[A1, B1], outputs=[out])
	# model.add(hub_layer)
	# model.add(tf.keras.layers.Dense(16, activation='relu'))
	# # model.add(tf.keras.layers.Dense( 4, activation='relu' ))
	# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

	# # inputB = keras.Input()
	# # x = keras.Dense(8, activation="relu")(inputB)

	# # prints a summary of the layers of the NN
	# model.summary()

	# model.compile(optimizer='adam',
    #           loss='binary_crossentropy',
    #           metrics=['accuracy'])

	# set aside 10000 comments and labels for validation
	# x_val = train_comments[:10000]
	# partial_x_train = train_comments[10000:]
	# y_val = train_labels[:10000]
	# partial_y_train = train_labels[10000:]

	# train the model
	# low number of epochs for testing


    print("starting keras")
	
	#model used to create embeddings for sentences
	# model = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
	# this one takes too long - 20mins per epoch
	# model = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1"
	
	#embeddings to use for words - 20-dimensional
	embeddings = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1"
	# input layer
	# hub_layer = hub.KerasLayer(embeddings, output_shape=[20], input_shape=[], dtype=tf.string, trainable=True, name="comment_in")
	# hub_layer_subreddit = hub.KerasLayer(embeddings, output_shape=[20], input_shape= [], dtype=tf.string, trainable=True, name="subreddit_in")

	embeddings = hub.KerasLayer(embeddings)

	embedding_layer_c = keras.layers.Embedding(input_dim=embeddings.shape[0],
                              output_dim=embeddings.shape[1], 
                              weights=[embeddings], 
                              trainable=False)

	input_c = keras.layers.InputLayer( )(embedding_layer_c)
	input_s = keras.layers.InputLayer(  )(embedding_layer_c)

	C1 = keras.layers.Dense(16, activation='relu')(input_c)

	S1 = keras.layers.Dense(16, activation='relu')(input_s)

	merge_layer = keras.layers.Concatenate( axis=-1 )( [ C1, S1 ] )
	hidden_layer = keras.layers.Dense( 4, activation='relu' )(merge_layer)
	output_layer = keras.layers.Dense(1, activation='sigmoid')(hidden_layer)

	final_model = keras.models.Model( inputs=[input_c, input_s ], outputs=output_layer)

	print( final_model.summary() )

	history = final_model.fit( [ train_comments, subreddit_train ], [train_labels]
                    ,
                    epochs=6,
                    batch_size=512,
                    validation_data=(test_comments, test_labels),
                    verbose=1)
	
	# # used for plotting

	# history_dict = history.history
	# history_dict.keys()

	# acc = history_dict['accuracy']
	# val_acc = history_dict['val_accuracy']
	# loss = history_dict['loss']
	# val_loss = history_dict['val_loss']

	# epochs = range(1, len(acc) + 1)

	# # "bo" is for "blue dot"
	# plt.plot(epochs, loss, 'bo', label='Training loss')
	# # b is for "solid blue line"
	# plt.plot(epochs, val_loss, 'b', label='Validation loss')
	# plt.title('Training and validation loss')
	# plt.xlabel('Epochs')
	# plt.ylabel('Loss')
	# plt.legend()

	# plt.show()

	# plt.clf()   # clear figure

	# plt.plot(epochs, acc, 'bo', label='Training acc')
	# plt.plot(epochs, val_acc, 'b', label='Validation acc')
	# plt.title('Training and validation accuracy')
	# plt.xlabel('Epochs')
	# plt.ylabel('Accuracy')
	# plt.legend()

	# plt.show()
