test_res = test_data

for _, w in enumerate(test_res):
    inputs = createInputs(w)
    out, _ = rnn.forward(inputs)
    res = softmax(out)<.5
    res = bool(res[0])
    test_res[w] = res
    
    if res:
        print(Fore.GREEN + w)
    else:
        print(Fore.RED + w)