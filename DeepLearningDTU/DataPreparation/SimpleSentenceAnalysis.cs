using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace ConsoleApplication1
{
    class SimpleSentenceAnalysis
    {
        static void Main(string[] args)
        {
            var sentences = File.ReadAllLines(@"C:\tmp\ExampleSentences.txt");
            var lixNumbers = new Dictionary<String, int>();
            foreach (var sentence in sentences)
            {
                var lixNumber = DetermineLixNumber(sentence);
                lixNumbers.Add(sentence, lixNumber);
            }
        }

        /// <summary>
        /// Calculates a lix number for the given sentence. Please note that a valid sentence must finish with a punctuation.
        /// </summary>
        public static int DetermineLixNumber(string sentence)
        {
            //(antal ord/antal punktummer) + (lange ord * 100/antal ord) (lange ord er længere end 7 bogstaver)
            var words = sentence.Split(' ');

            var numberOfWords = words.Length;
            var numberOfPunctuations = words.Where(word => word.Contains('.')).Count();
            var numberOfLongWords = words.Where(word => word.Length > 7).Count();

            var lixNumber = (numberOfWords + 0.0) / numberOfPunctuations + (numberOfLongWords * 100) / numberOfWords;
            return (int)lixNumber;
        }

        /// <summary>
        /// Returns the number of words in the sentence.
        /// </summary>
        public static int NumberOfWords(string sentence)
        {
            var words = sentence.Split(' ');
            return words.Length;
        }

        /// <summary> Returns the number of commas in the sentence </summary>
        public static int NumberOfCommas(string sentence)
        {
            return sentence.Where(c => c == ',').Count();
        }

        /// <summary>
        /// Determines if the sentence is suitable - using a predefined set of rules.
        /// </summary>
        public static bool IsSentenceSuitable(string sentence)
        {
            // number of words must be less than or equal to ten
            if (NumberOfWords(sentence) > 10)
                return false;
            if (NumberOfCommas(sentence) != 1)
                return false;

            return true;
        }

        /// <summary>
        /// Sanity checks of the sentence:
        /// - The sentence should contain at least 3 words, in order to accept it.
        /// - The sentence cannot be longer than 30? words
        /// - 
        /// </summary>
        public static bool IsSentenceReasonable(string sentence)
        {
            return true;
        }
    }
}
