# -*- coding: utf-8 -*-
import artm.messages_pb2, artm.library, sys, glob

# Parse collection
data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ''
target_folder = 'test'
collection_name = 'test'

batches_found = len(glob.glob(target_folder + "/*.batch"))
if batches_found == 0:
  print "No batches found, parsing them from textual collection...",
  collection_parser_config = artm.messages_pb2.CollectionParserConfig();
  collection_parser_config.format = artm.library.CollectionParserConfig_Format_BagOfWordsUci

  collection_parser_config.docword_file_path = data_folder + 'docword.'+ collection_name + '.txt'
  collection_parser_config.vocab_file_path = data_folder + 'vocab.'+ collection_name + '.txt'
  collection_parser_config.target_folder = target_folder
  collection_parser_config.dictionary_file_name = 'dictionary'
  unique_tokens = artm.library.Library().ParseCollection(collection_parser_config);
  print " OK."
else:
  print "Found " + str(batches_found) + " batches, using them."
  unique_tokens  = artm.library.Library().LoadDictionary(target_folder + '/dictionary');

# Create master component and infer topic model
with artm.library.MasterComponent() as master:
  master.config().cache_theta = True
  master.Reconfigure()
  # Create dictionary with tokens frequencies
  dictionary           = master.CreateDictionary(unique_tokens)
  
  # Create one top-token score per each class_id
  ru_top_tokens_score = master.CreateTopTokensScore(num_tokens = 30, class_id='@default_class')
  en_top_tokens_score = master.CreateTopTokensScore(class_id='@labels')

  # Configure basic scores
  perplexity_score     = master.CreatePerplexityScore(name='my_perplexity_score')
  
  perplexity_score_config = artm.messages_pb2.PerplexityScoreConfig();
  perplexity_score_config.dictionary_name='dictionary'
  perplexity_score_config.stream_name='@labels'
  
  theta_snippet_score = master.CreateThetaSnippetScore()

  # Populate class_id and class_weight in ModelConfig
  config = artm.messages_pb2.ModelConfig()
  config.class_id.append('@default_class')
  config.class_weight.append(0.10)
  config.class_id.append('@labels')
  config.class_weight.append(1.00)

  # Configure the model
  model = master.CreateModel(topics_count = 15, inner_iterations_count = 10, config=config)
  model.Initialize(dictionary)       # Setup initial approximation for Phi matrix.
  model.EnableScore(perplexity_score)
  model.EnableScore(ru_top_tokens_score)
  model.EnableScore(en_top_tokens_score)
  model.EnableScore(theta_snippet_score)

  for iter in range(0, 8):
    master.InvokeIteration(disk_path=target_folder)  # Invoke one scan over all batches,
    master.WaitIdle();                               # and wait until it completes.
    model.Synchronize();                             # Synchronize topic model.
   
  artm.library.Visualizers.PrintTopTokensScore(ru_top_tokens_score.GetValue(model))
  artm.library.Visualizers.PrintTopTokensScore(en_top_tokens_score.GetValue(model))
  
  ps=perplexity_score.GetValue(model)
  f = open('perp_15topics.csv', 'w')
  f.write('value '+str(ps.value)+'\n') #A perplexity value which is calculated as exp(-raw/normalizer).
  f.write('raw '+str(ps.raw)+'\n') #A numerator of perplexity calculation. This value is equal to the likelihood of the topic model.
  f.write('normalizer '+str(ps.normalizer)+'\n') #A denominator of perplexity calculation. This value is equal to the total number of tokens in all processed items.
  f.write('zero_words '+str(ps.zero_words)+'\n') # number of tokens that have zero probability p(w|t,d) in a document. Such tokens are evaluated based on to unigram document model or unigram colection model.
  f.write('theta_sparsity_value '+str(ps.theta_sparsity_value)+'\n') #A fraction of zero entries in the theta matrix.
  f.write('theta_sparsity_zero_topics '+str(ps.theta_sparsity_zero_topics)+'\n')
  f.write('theta_sparsity_total_topics  '+str(ps.theta_sparsity_total_topics )+'\n')
  f.close()
 
  theta_matrix = master.GetThetaMatrix(model, clean_cache=True)
  print "Option 2. Full ThetaMatrix cached during last iteration, #items = %i" % len(theta_matrix.item_id)

  
  f = open('theta_test.csv', 'w')
  for j, item in enumerate(theta_matrix.item_weights):
	  str1=''
	  docvector=[]
	  for val in item.value:
		  docvector.append(val) 
		  str1=str1+str(val)+';'
	  label=docvector.index(max(docvector))+1 #index(x) Возвращает наименьшее i, такое, что s[i] == x.
	  str1=str1+str(label)+';'
	  f.write(str1+'\n')
  f.close()      
	

