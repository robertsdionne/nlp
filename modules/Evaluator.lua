require 'torch'
require 'nn'
dofile "tags.lua"

local Evaluator = torch.class('nn.Evaluator')

function Evaluator:__init()
end

function Evaluator:evaluateTagger(pos_tagger, tagged_sentences, training_vocabulary, verbose, plot)
  -- Ported directly from POSTaggerTester.java from the assignments.
  local num_tags = 0.0
  local num_tags_correct = 0.0
  local num_unknown_words = 1e-8
  local num_unknown_words_correct = 1e-8
  local num_decoding_inversions = 0
  local sentenceAccHis = {}
  local sentenceLenHis = torch.zeros(200)
  local sentenceCountHis = torch.zeros(200)
  local wordCorrect = {}
  local wordCount = {}
  local errPosHis = {}
  local confusion = {}
  local toTags = {}
  local toCount = 0
  --init the confusion matrix
  for i = 1, #tags do 
    for j = 1, #tags do
      confusion[tags[i].." "..tags[j]] = 0
    end
  end
  

  for i = 1, #tagged_sentences do
    local sentenceAcc = 0
    if i % 100 == 0 and verbose then
      print("finished "..i.."sentences / "..#tagged_sentences);
    end
    local tagged_sentence = tagged_sentences[i]
    local words = tagged_sentence.words
    local gold_tags = tagged_sentence.tags
    local guessed_tags = pos_tagger:tag(tagged_sentence)
    for position = 1, #words do
      local word = words[position]
      local gold_tag = gold_tags[position]
      local guessed_tag = guessed_tags[position]
      -- print("Gold: "..gold_tag.."   Guess: "..guessed_tag);
      if wordCount[word] == nil then
        wordCount[word] = {count = 1, correct = 0, tagp = {}}
      else 
        wordCount[word].count = wordCount[word].count + 1
      end
      if guessed_tag == gold_tag then
        sentenceAcc = sentenceAcc + 1.0
        num_tags_correct = num_tags_correct + 1.0
        -- ignore NN->NNP NNP->NN
        wordCount[word].correct = wordCount[word].correct + 1
        
        if word == "to" then
          table.insert(toTags, gold_tag)
        end
      else
        table.insert(wordCount[word].tagp, gold_tag.."->"..guessed_tag)
        if (guessed_tag == "NN" and gold_tag == "NNP") or (gold_tag == "NN" and guessed_tag == "NNP") then
          num_tags_correct = num_tags_correct + 1.0
        end
        table.insert(errPosHis, position / #words)
        confusion[gold_tag .. " " .. guessed_tag] = confusion[gold_tag .. " " .. guessed_tag] + 1
      end
      --record the confusion mat
      if word == "to" then
        toCount = toCount + 1
      end

      num_tags = num_tags + 1.0
      if not training_vocabulary[word] then
        -- print(word .. ' ' .. guessed_tag .. ' ' .. gold_tag)
        if guessed_tag == gold_tag then
          num_unknown_words_correct = num_unknown_words_correct + 1.0
        end
        num_unknown_words = num_unknown_words + 1.0
      end
    end
    -- score_of_gold_tagging = pos_tagger:scoreTagging(tagged_sentence)
    -- score_of_guessed_tagging = pos_tagger:scoreTagging(nn.TaggedSentence(words, guessed_tags))
    -- if score_of_gold_tagging > score_of_guessed_tagging then
    --   num_decoding_inversions = num_decoding_inversions + 1
    --   if verbose then
    --     print('WARNING: Decoder suboptimality detected. ' ..
    --         'Gold tagging has higher score than guessed tagging.')
    --   end
    -- end
    if verbose then
      print(self:alignedTaggings(words, gold_tags, guessed_tags, true))
    end
    sentenceAcc = sentenceAcc / #words
    table.insert(sentenceAccHis, sentenceAcc)
    --table.insert(sentenceLenHis, #words)
    sentenceLenHis[#words] = sentenceLenHis[#words] + sentenceAcc
    sentenceCountHis[#words] = sentenceCountHis[#words] + 1
    -- if (1-sentenceAcc) * #words > 10 then
    --   for wn = 1, #words do
    --   print(tagged_sentence.words[wn]..'  '..tagged_sentence.tags[wn]..'  '..guessed_tags[wn])
    --   end
    --   io.read()
    -- end
  end

  for k,v in pairs(wordCount) do
    v.rate = v.correct / v.count
    v.word = k
    if v.count > 5 then
      table.insert(wordCorrect, v)
    end
  end
  table.sort(wordCorrect, function (a,b) return a.rate<b.rate end)
  for ii = 1, 50 do
    print(wordCorrect[ii])
  end
  -- gnuplot.hist(torch.Tensor(sentenceAccHis),40)
  -- --io.read()
  -- gnuplot.hist(torch.Tensor(errPosHis),40)
  -- gnuplot.xlabel("Related Position")
  -- gnuplot.ylabel("Number of Mistakes")
  --io.read()
  --init the confusion matrix
  -- for i = 1, #tags do 
  --   io.write('{')
  --   for j = 1, #tags do
  --     io.write(confusion[tags[i].." "..tags[j]])
  --     if j~=#tags then
  --       io.write(',') 
  --     else
  --       io.write('},\n')
  --     end
  --   end
  -- end
  -- print(toTags)
  -- print(#toTags/toCount)
  -- print(sentenceCountHis)
  -- print(torch.cdiv(sentenceLenHis,sentenceCountHis))
  -- io.read()
  print('  Tag Accuracy: ' .. (num_tags_correct / num_tags))
  print(num_tags)
  print(num_tags_correct)
  print(num_tags)
  print('  (Unknown Accuracy: ' .. (num_unknown_words_correct / num_unknown_words) .. ')')
  -- print('  Decoder Suboptimalities Detected: ' .. num_decoding_inversions)
end

function Evaluator:alignedTaggings(words, gold_tags, guessed_tags, suppress_correct_tags)
  -- TODO(robertsdionne): port from POSTaggerTester.java from the assignments.
  return ''
end
