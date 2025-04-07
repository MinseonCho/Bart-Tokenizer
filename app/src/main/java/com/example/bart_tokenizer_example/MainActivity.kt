package com.example.bart_tokenizer_example

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.BasicAlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.DialogProperties
import com.example.bart_tokenizer_example.bart.tokenizer.BartTokenizer
import com.example.bart_tokenizer_example.bart.tokenizer.TestData
import com.example.bart_tokenizer_example.ui.theme.RoBERTaTokenizerExampleTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.LongBuffer

private const val MAX_GENERATION_LENGTH = 256
private const val BUFFER_SIZE = 8192
private const val START_TOKEN_ID = 0L
private const val END_TOKEN_ID = 2L

class MainActivity : ComponentActivity() {
    private val tokenizer = BartTokenizer()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        tokenizer.initialize(this)
        setContent {
            RoBERTaTokenizerExampleTheme {
                MainScreen(tokenizer = tokenizer)
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun MainScreen(
    tokenizer: BartTokenizer,
    modifier: Modifier = Modifier,
) {
    var inputText by remember { mutableStateOf(TestData.text6) }
    var isProgressBarVisible by remember { mutableStateOf(false) }
    var isDialogVisible by remember { mutableStateOf(false) }
    var originalText by remember { mutableStateOf("") }
    var summaryResult by remember { mutableStateOf("") }
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()

    Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            Column(modifier = modifier.padding(16.dp)) {
                OutlinedTextField(
                    value = inputText,
                    onValueChange = { inputText = it },
                    label = { Text("요약할 문장을 입력하세요") },
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                )
                Spacer(modifier = Modifier.height(16.dp))
                Button(
                    onClick = {
                        isProgressBarVisible = true
                        originalText = inputText

                        coroutineScope.launch(Dispatchers.IO) {
                            summaryResult = summarize(
                                text = inputText,
                                context = context,
                                tokenizer = tokenizer
                            )
                            isDialogVisible = true
                            isProgressBarVisible = false
                        }
                    },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(48.dp)
                ) {
                    Text(
                        text = "요약하기",
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            if (isProgressBarVisible) {
                CircularProgressIndicator(
                    modifier = Modifier.align(Alignment.Center)
                )
            }
        }

        if (isDialogVisible) {
            BasicAlertDialog(
                onDismissRequest = { isDialogVisible = false },
                properties = DialogProperties(
                    usePlatformDefaultWidth = false
                )
            ) {
                Surface(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(24.dp),
                    shape = MaterialTheme.shapes.medium,
                    tonalElevation = 6.dp
                ) {
                    Column(modifier = Modifier.padding(20.dp)) {
                        Text(
                            text = "요약 결과",
                            style = MaterialTheme.typography.titleMedium
                        )
                        Spacer(modifier = Modifier.height(12.dp))
                        Text(
                            text = summaryResult,
                            style = MaterialTheme.typography.bodyMedium
                        )
                        Spacer(modifier = Modifier.height(20.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween
                        ) {
                            Text(
                                text = "원본 길이: ${originalText.length}\n요약 길이: ${summaryResult.length}"
                            )
                            Button(
                                onClick = { isDialogVisible = false },
                                modifier = Modifier
                            ) {
                                Text("닫기")
                            }
                        }
                    }
                }
            }
        }
    }
}

private fun summarize(
    context: Context,
    text: String,
    tokenizer: BartTokenizer
): String {
    val tokens = tokenizer.tokenize(text)
    println("mscho, Tokens: $tokens")
    val inputIds = tokenizer.convertTokensToIds(tokens).map { it }.toLongArray()
    println("mscho, Input IDs: ${inputIds.contentToString()}")

    val attentionMask = LongArray(inputIds.size) { 1 }
    val env = OrtEnvironment.getEnvironment()
    val encoderSession = createSession(context, env, "encoder_model_q4.onnx")
    val decoderSession = createSession(context, env, "decoder_model_q4.onnx")

    val encoderHiddenStates = runEncoder(env, encoderSession, inputIds, attentionMask)
    val generatedSequence = generateSequence(env, decoderSession, encoderHiddenStates, attentionMask)
    
    return tokenizer.decode(generatedSequence).also {
        println("mscho, 요약 결과: $it")
    }
}

private fun runEncoder(
    env: OrtEnvironment,
    encoderSession: OrtSession,
    inputIds: LongArray,
    attentionMask: LongArray
): Array<Array<FloatArray>> {
    val inputShape = longArrayOf(1, inputIds.size.toLong())
    val inputTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), inputShape)
    
    val attentionShape = longArrayOf(1, attentionMask.size.toLong())
    val attentionTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), attentionShape)

    val encoderInputs = mapOf(
        "input_ids" to inputTensor,
        "attention_mask" to attentionTensor
    )

    return encoderSession.run(encoderInputs)[0].value as Array<Array<FloatArray>>
}

private fun generateSequence(
    env: OrtEnvironment,
    decoderSession: OrtSession,
    encoderHiddenStates: Array<Array<FloatArray>>,
    attentionMask: LongArray
): List<Long> {
    val encoderHiddenShape = longArrayOf(
        1,
        attentionMask.size.toLong(),
        encoderHiddenStates[0][0].size.toLong()
    )
    
    val generated = mutableListOf<Long>()
    generated.add(START_TOKEN_ID)

    for (i in 0 until MAX_GENERATION_LENGTH) {
        val decoderInputIds = generated.toLongArray()
        val decoderShape = longArrayOf(1, decoderInputIds.size.toLong())
        
        val decoderInputs = mapOf(
            "encoder_attention_mask" to OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(attentionMask),
                longArrayOf(1, attentionMask.size.toLong())
            ),
            "input_ids" to OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(decoderInputIds),
                decoderShape
            ),
            "encoder_hidden_states" to OnnxTensor.createTensor(
                env,
                flattenHiddenStates(encoderHiddenStates),
                encoderHiddenShape
            ),
        )

        val logits = decoderSession.run(decoderInputs)[0].value as Array<Array<FloatArray>>
        val nextTokenId = predictNextToken(logits[0].last())

        println("mscho, Generated token ID: $nextTokenId")
        if (nextTokenId == END_TOKEN_ID) {
            println("mscho, End token generated, stopping generation")
            break
        }
        generated.add(nextTokenId)
    }
    
    println("mscho, Generated sequence: $generated")
    return generated
}

private fun createSession(
    context: Context,
    env: OrtEnvironment,
    modelName: String
): OrtSession {
    return context.assets.open(modelName).use { modelStream ->
        val size = modelStream.available()
        val buffer = ByteBuffer.allocateDirect(size)
        val bytes = ByteArray(BUFFER_SIZE)
        var read: Int
        while (modelStream.read(bytes).also { read = it } != -1) {
            buffer.put(bytes, 0, read)
        }
        buffer.rewind()
        env.createSession(buffer)
    }
}

/**
 *  Greedy Decoding
 * 	•	매 시점마다 확률이 가장 높은 토큰을 1개 선택
 * 	•	가장 단순하고 빠른 방법
 *
 * 	Top-k Sampling
 * 	•	확률 상위 k개 토큰 중 무작위 선택
 * 	•	다양성 있는 결과 생성
 *
 * 	Top-p (Nucleus) Sampling
 * 	•	누적 확률이 p 이상이 될 때까지 토큰을 포함하고, 그 중 무작위 선택
 * 	•	top-k보다 더 유연하고 다양성 보장
 *
 * 	등.. 방법이 있고 greedy 방식 채택
 */
private fun predictNextToken(logits: FloatArray): Long {
    return logits.indices.maxByOrNull { logits[it] }?.toLong() ?: 0L
}

private fun flattenHiddenStates(encoderOutput: Array<Array<FloatArray>>): FloatBuffer {
    val batchSize = encoderOutput.size
    val seqLength = encoderOutput[0].size
    val hiddenSize = encoderOutput[0][0].size

    val flatBuffer = FloatBuffer.allocate(batchSize * seqLength * hiddenSize)

    for (batch in 0 until batchSize) {
        for (seq in 0 until seqLength) {
            for (hidden in 0 until hiddenSize) {
                flatBuffer.put(encoderOutput[batch][seq][hidden])
            }
        }
    }

    flatBuffer.rewind()
    return flatBuffer
}

// 토크나이저 테스트 용
fun bart(tokenizer: BartTokenizer) {
    val tokens = tokenizer.tokenize(
        "Unmaintainable codebases often lead to technical debt accumulation.😵"
    )
    val ids = tokenizer.convertTokensToIds(tokens)
    val result = tokenizer.decode(ids)

    println("mscho, tokens: $tokens")
    println("mscho, ids: $ids")
    println("rmscho, result: $result")
}
