package com.example.bart_tokenizer_example.bart.tokenizer

import android.content.Context
import com.google.gson.Gson
import com.google.gson.reflect.TypeToken
import java.io.BufferedReader
import java.nio.charset.StandardCharsets

class BartTokenizer {
    // 필수 파일
    private val vocabFileName = "vocab.json"
    private val mergesFileName = "merges.txt"

    // Special token 정의
    private val bosToken: String = "<s>"
    private val eosToken: String = "</s>"
    private val padToken: String = "<pad>"
    private val unkToken: String = "<unk>"

    // vocab 및 merge 파일 로딩 후 저장될 맵
    private lateinit var encoder: Map<String, Long>  // token, id
    private lateinit var decoder: Map<Long, String>  // id, token
    private lateinit var bpeRanks: Map<Pair<String, String>, Int>  // BPE merge 순서

    // byte-level encoding/decoding 테이블
    private val byteEncoder: Map<Byte, String> = bytesToUnicode()
    private val byteDecoder: Map<String, Byte> = byteEncoder.entries.associate { it.value to it.key }

    // BPE 결과 캐싱용
    private val cache = mutableMapOf<String, String>()

    // 토큰 패턴 정의 (BART의 기본 패턴과 동일)
    private val tokenPattern = Regex("""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    // vocab 및 merges 파일 내용을 읽어서 셋팅
    fun initialize(context: Context) {
        encoder = loadVocabFromAssets(context)
        decoder = encoder.entries.associate { it.value to it.key }
        bpeRanks = loadMergesFromAssets(context)
    }

    // 입력 문자열을 BPE 토큰 목록으로 변환
    fun tokenize(
        text: String,
        addPrefixSpace: Boolean = false,
    ): List<String> {
        val preparedText = if (addPrefixSpace) {
            prepareForTokenization(text)
        } else {
            text
        }
        val tokens = mutableListOf<String>()

        for (word in tokenPattern.findAll(preparedText)) {
            val encoded = word.value
                .toByteArray(Charsets.UTF_8)
                .joinToString("") { byteEncoder[it] ?: "" }  // 바이트를 유니코드로 인코딩
            tokens += bpe(encoded).split(" ")  // BPE 적용
        }

        return tokens
    }

    // 토큰을 id로 변환 + special token 붙이기
    fun convertTokensToIds(tokens: List<String>): List<Long> {
        val tokenIds = tokens.map { encoder[it] ?: encoder[unkToken]!! }
        return buildInputIdsWithSpecialToken(tokenIds)
    }

    // id -> 토큰
    private fun convertIdsToTokens(ids: List<Long>): List<String> {
        return ids.map { decoder[it] ?: unkToken }
    }

    // BPE 토큰들을 문자열로 합치고 UTF-8로 복호화
    private fun convertTokensToString(tokens: List<String>): String {
        val merged = tokens.joinToString("") // ex: "ĠReadingĠbedtime..."
        val byteChars = merged.map {
            byteDecoder[it.toString()] ?: '?'.code.toByte()
        }
        return byteChars.toByteArray().toString(Charsets.UTF_8).trim()
    }

    // 최종적으로 디코딩된 자연어 문자열 리턴
    fun decode(ids: List<Long>): String {
        val tokens = convertIdsToTokens(ids)
        val merged = convertTokensToString(tokens)

        // special token 제거
        return merged
            .replace(bosToken, "")
            .replace(eosToken, "")
            .replace(padToken, "")
            .trim()
    }

    // input_ids 앞뒤에 special token 삽입: <s> ... </s>
    private fun buildInputIdsWithSpecialToken(inputIds: List<Long>): List<Long> {
        return buildList {
            add(getClsTokenId())
            addAll(inputIds)
            add(getEosTokenId())
        }
    }

    // BART의 특성상, 문장 앞에 공백을 붙이는 경우가 많음
    private fun prepareForTokenization(text: String): String {
        return if (text.isNotEmpty() && !text.first().isWhitespace()) {
            " $text"
        } else {
            text
        }
    }

    // 핵심: Byte-Pair Encoding 적용
    private fun bpe(token: String): String {
        if (cache.containsKey(token)) return cache[token]!!

        var word = token.map { it.toString() } // there, "t", "h", "e", "r", "e"
        var pairs = getPairs(word) // (t, h) (h, e)... (t, he) (he, r)

        while (pairs.isNotEmpty()) {
            val bigram = pairs.minByOrNull {
                bpeRanks[it] ?: Int.MAX_VALUE // 병합할 bigram 중 BPE merge 우선순위가 가장 높은 것을 선택
            } ?: break
            if (bpeRanks.containsKey(bigram).not()) break

            val (first, second) = bigram
            val newWord = mutableListOf<String>()
            var i = 0

            while (i < word.size) {
                val j = findIndexOf(word, first, i)
                if (j == -1) {
                    newWord.addAll(word.subList(i, word.size))
                    break
                }

                newWord.addAll(word.subList(i, j))
                if (j < word.size - 1 && word[j + 1] == second) {
                    newWord.add(first + second)
                    i = j + 2
                } else {
                    newWord.add(word[j])
                    i = j + 1
                }
            }

            word = newWord
            pairs = getPairs(word)
        }

        val result = word.joinToString(" ")
        cache[token] = result
        return result
    }

    // 리스트에서 target 문자열을 찾는 helper
    private fun findIndexOf(list: List<String>, target: String, start: Int): Int {
        for (i in start until list.size) {
            if (list[i] == target) return i
        }
        return -1
    }

    // 인접 쌍들을 추출하는 함수: [(a,b), (b,c), (c,d)] 이런 식으로
    private fun getPairs(word: List<String>): Set<Pair<String, String>> {
        return word.zipWithNext().toSet()
    }

    // 바이트 → 유니코드 매핑을 위한 테이블 생성
    private fun bytesToUnicode(): Map<Byte, String> {
        val bs = mutableListOf<Int>()
        bs += ('!'.code..'~'.code)
        bs += ('¡'.code..'¬'.code)
        bs += ('®'.code..'ÿ'.code)

        val cs = bs.toMutableList()
        var n = 0
        for (b in 0..255) {
            if (b !in bs) {
                bs += b
                cs += 256 + n
                n += 1
            }
        }

        val unicodeMap = mutableMapOf<Byte, String>()
        for (i in bs.indices) {
            unicodeMap[bs[i].toByte()] = cs[i].toChar().toString()
        }
        return unicodeMap
    }

    // vocab.json 로드
    private fun loadVocabFromAssets(context: Context): Map<String, Long> {
        val json = context.assets.open(vocabFileName)
            .bufferedReader(StandardCharsets.UTF_8)
            .use(BufferedReader::readText)
        val type = object : TypeToken<Map<String, Long>>() {}.type

        return Gson().fromJson(json, type)
    }

    // merges.txt 로드
    private fun loadMergesFromAssets(context: Context): Map<Pair<String, String>, Int> {
        val lines = context.assets.open(mergesFileName)
            .bufferedReader(StandardCharsets.UTF_8)
            .readLines()

        return lines
            .dropWhile { it.startsWith("#") || it.isBlank() }
            .mapIndexedNotNull { index, line ->
                val parts = line.trim().split(" ")
                if (parts.size == 2) {
                    Pair(parts[0], parts[1]) to index
                } else {
                    null
                }
            }.toMap()
    }

    // <s> 토큰의 ID
    private fun getClsTokenId(): Long {
        return encoder[bosToken] ?: error("bosToken not in vocab")
    }

    // </s> 토큰의 ID
    private fun getEosTokenId(): Long {
        return encoder[eosToken] ?: error("eosToken not in vocab")
    }
}
