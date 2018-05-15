def initialize():
    global train_data,val_data,test_data
    def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data
    corpus = data.Corpus(args.data)
    eval_batch_size = 10
    train_data = batchify(corpus.train, args.batch_size)
    val_data = batchify(corpus.valid, eval_batch_size)
    test_data = batchify(corpus.test, eval_batch_size)


initialize()
