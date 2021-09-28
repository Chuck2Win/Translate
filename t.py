# candidates - batch size, num_beams, 2(list, score)
# single beam search 

# single beam search



i = 0
with torch.no_grad():
    model.eval()
    outputs = model.forward(encoder_input_ids.to(args.device), decoder_input_ids.to(args.device)) # bs, seq_len_2, n_vocab
for b in range(len(outputs)): # batch
    c  = outputs[b][i].topk(num_beams,dim=-1)
    for k in range(num_beams): # num_beams
        candidates[b][k][0].append(c.indices[k].item())
        candidates[b][k][1]+=(c.values[k].item())

def make_batch(candidates):
    # candidates shape : (batch size, num_beams, 2) ~ list, log prob
    result = []
    for i in range(num_beams):
        batch = encoder_input_ids.new(encoder_input_ids.shape).fill_(tokenizer.pad_token_id).tolist()
        length = len(candidates[0][0][0])
        for b  in range(len(candidates)):
            batch[b][1:length+1] = candidates[b][i][0]
        result.append(batch)
    return result
# x ~ batch size, num_beams, seq_len
# y ~ batch size, num_beams, seq_len
x = make_batch(candidates)
x = torch.LongTensor(x).transpose(0,1).reshape(-1, max_length).to(args.device)
y = encoder_input_ids.unsqueeze(1).repeat(1,num_beams,1).reshape(-1,args.seq_len_1) # batch_size*num_beams, max_length

for i in range(1,max_length):
    with torch.no_grad():
        model.eval()
        outputs = model.forward(y.to(args.device), x.to(args.device)) # bs, seq_len_2, n_vocab
    # outputs ~ batch size * num_beams, seq_len, n_vocab
    outputs = outputs.reshape(args.batch_size, num_beams, max_length, -1) # batch size, num_beams, seq_len, n_vocab
    for b in range(args.batch_size): # batch
        c  = outputs[b,:,i,:].topk(num_beams,dim=-1)
        for k in range(num_beams): # num_beams
            candidates[b][k][0].append(c.indices[k].item())
            candidates[b][k][1]+=(c.values[k].item())
outputs
