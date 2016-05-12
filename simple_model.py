import artm.messages_pb2, artm.library, sys, glob

# Parse collection
data_folder = sys.argv[1] if (len(sys.argv) >= 2) else ''
target_folder = 'kos'
collection_name = 'kos'

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
  # Create dictionary with tokens frequencies
  dictionary           = master.CreateDictionary(unique_tokens)

  # Configure basic scores
  perplexity_score     = master.CreatePerplexityScore(name='my_perplexity_score')
  #sparsity_theta_score = master.CreateSparsityThetaScore()
  #sparsity_phi_score   = master.CreateSparsityPhiScore()
  top_tokens_score     = master.CreateTopTokensScore(num_tokens = 30)
  theta_snippet_score = master.CreateThetaSnippetScore()
  
  perplexity_score_config = artm.messages_pb2.PerplexityScoreConfig();
  perplexity_score_config.dictionary_name='dictionary'

  # Configure basic regularizers
  smsp_theta_reg   = master.CreateSmoothSparseThetaRegularizer()
  smsp_phi_reg     = master.CreateSmoothSparsePhiRegularizer()
  decorrelator_reg = master.CreateDecorrelatorPhiRegularizer()

  # Configure the model
  model = master.CreateModel(topics_count = 15, inner_iterations_count = 10)
  model.EnableScore(perplexity_score)
  model.EnableScore(top_tokens_score)
  model.EnableScore(theta_snippet_score)
  model.Initialize(dictionary)       # Setup initial approximation for Phi matrix.

  for iter in range(0, 8):
    master.InvokeIteration(disk_path=target_folder)  # Invoke one scan over all batches,
    master.WaitIdle();                               # and wait until it completes.
    model.Synchronize();                             # Synchronize topic model.
   
  artm.library.Visualizers.PrintTopTokensScore(top_tokens_score.GetValue(model))
  ts=theta_snippet_score.GetValue(model)
  #artm.library.Visualizers.PrintThetaSnippetScore(ts) 
  
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
  
  f = open('text.csv', 'w')
  for item in theta_matrix.item_weights:
      str1=''
      for val in item.value:
        str1=str1+str(val)+';'
      f.write(str1 + '\n')
  f.close()
  
  
 
