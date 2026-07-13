# Feature Dictionary

## Text-surface features

These features are calculated from Management Discussion and Analysis (MDA) text only.
They are available for train/validation and test records and do not use the fraud label.

| Feature | Description |
| --- | --- |
| `mda_char_count` | Number of characters in the disclosure. |
| `mda_word_count` | Number of alphabetic words in the disclosure. |
| `mda_sentence_count` | Number of detected sentences. |
| `mda_avg_sentence_words` | Average number of words per sentence. |
| `mda_log_word_count` | Natural log of one plus the word count. |
| `mda_flesch_reading_ease` | Flesch Reading Ease score; higher values indicate easier text. |
| `mda_flesch_kincaid_grade` | Estimated education grade required to read the text. |
| `mda_gunning_fog` | Gunning Fog readability index. |
| `mda_complex_word_ratio` | Share of words estimated to have at least three syllables. |
| `mda_digit_ratio` | Share of characters that are digits. |
| `mda_punctuation_ratio` | Share of characters that are punctuation. |
| `mda_uppercase_ratio` | Share of alphabetic characters that are uppercase. |
| `mda_avg_word_length` | Average number of characters per word. |
| `mda_long_sentence_ratio` | Share of sentences with at least 25 words. |
| `mda_lexical_diversity` | Unique case-insensitive words divided by total words. |
| `mda_text_available` | One when usable MDA text exists; zero when it is missing or blank. |

When MDA text is unavailable, all text-surface features except `mda_text_available`
are missing. Model preprocessing handles those values consistently with the
existing LM features.
