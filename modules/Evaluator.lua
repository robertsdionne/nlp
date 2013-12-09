require 'torch'
require 'nn'

local Evaluator = torch.class('nn.Evaluator')

function Evaluator:__init()
end

function Evaluator:evaluateTagger(pos_tagger, tagged_sentences, training_vocabulary, verbose)
  -- Ported directly from POSTaggerTester.java from the assignments.
  local num_tags = 0.0
  local num_tags_correct = 0.0
  local num_unknown_words = 1e-8
  local num_unknown_words_correct = 1e-8
  local num_decoding_inversions = 0
  for i = 1, #tagged_sentences do
    if i % 100 == 0 then
      print("finished "..i.."sentences / "..#tagged_sentences);
    end
    local tagged_sentence = tagged_sentences[i]
    local words = tagged_sentence.words
    local gold_tags = tagged_sentence.tags
    local guessed_tags = pos_tagger:tag(tagged_sentence)
    for position = 1, #words do
      local word = words[positon]
      local gold_tag = gold_tags[position]
      local guessed_tag = guessed_tags[position]
      -- print("Gold: "..gold_tag.."   Guess: "..guessed_tag);
      if guessed_tag == gold_tag then
        num_tags_correct = num_tags_correct + 1.0
      end
      num_tags = num_tags + 1.0
      if not training_vocabulary[word] then
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
  end
  print('  Tag Accuracy: ' .. (num_tags_correct / num_tags))
  print('  (Unknown Accuracy: ' .. (num_unknown_words_correct / num_unknown_words) .. ')')
  print('  Decoder Suboptimalities Detected: ' .. num_decoding_inversions)
end

function Evaluator:alignedTaggings(words, gold_tags, guessed_tags, suppress_correct_tags)
  -- TODO(robertsdionne): port from POSTaggerTester.java from the assignments.
  return ''
end
