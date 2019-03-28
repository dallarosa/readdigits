package me.dallarosa.readdigits

import android.content.Intent
import android.content.res.AssetManager
import android.graphics.*
import android.os.Bundle
import android.provider.MediaStore
import android.support.v7.app.AppCompatActivity
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel


class MainActivity : AppCompatActivity() {
    val REQUEST_IMAGE_CAPTURE = 1
    lateinit var myInterpreter: Interpreter
    private val DIM_BATCH_SIZE = 1

    private val DIM_PIXEL_SIZE = 4

    private val DIM_IMG_SIZE_X = 28
    private val DIM_IMG_SIZE_Y = 28

    private val NEGATIVE = floatArrayOf(
        -1.0f, 0f, 0f, 0f, 255f, // red
        0f, -1.0f, 0f, 0f, 255f, // green
        0f, 0f, -1.0f, 0f, 255f, // blue
        0f, 0f, 0f, 1.0f, 0f  // alpha
    )


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        var cameraButton = findViewById<Button>(R.id.camera_button)

        cameraButton.setOnClickListener( object : View.OnClickListener{
            override fun onClick(v: View?) {
                takePicture()
            }}
        )
        val assetManager = resources.assets
        myInterpreter =  Interpreter(loadModelFile(assetManager,"mnist_model.tflite"))

    }

    private fun takePicture() {
        Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { takePictureIntent ->
            takePictureIntent.resolveActivity(packageManager)?.also {
                startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent) {
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            val imageBitmap = data.extras.get("data") as Bitmap
            Log.v("ReadDigits", String.format("%d x %d",imageBitmap.width, imageBitmap.height))

            val imageView = findViewById<ImageView>(R.id.imageView)

            val numberImage = mnistizeImage(imageBitmap)

            imageView.setImageBitmap(numberImage)

            var labelProb: Array<FloatArray> = Array(1,{FloatArray(10)})
            
            val imgData = ByteBuffer.allocateDirect(
                DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
            val intValues = IntArray(28*28)
            numberImage.getPixels(intValues, 0,
                numberImage.getWidth(), 0, 0, numberImage.getWidth(), numberImage.getHeight())
            // Convert masked pixels to 0-255 scale.
            Log.v("Array", intValues.toString())
            var pixel = 0
            for (v in intValues) {
                //0.2989, 0.5870, 0.1140
                val postValue = if ((0.2989*(v shr 16 and 0xFF) +
                            0.5870*(v shr 8 and 0xFF) + 0.1140*(v and 0xFF)) > 127) 255 else 0
                imgData.putFloat(postValue.toFloat())
            }

            myInterpreter.run(imgData, labelProb)
            Log.v("Probability",labelProb.size.toString())
            Log.v("Probability",labelProb[0].size.toString())
            for(prob in labelProb[0]) {
                Log.v("Probability",prob.toString())
            }
            val msg = findViewById<TextView>(R.id.result_msg)
            val txt = findViewById<TextView>(R.id.result_txt)

            msg.visibility = TextView.VISIBLE
            txt.visibility = TextView.VISIBLE
            val maxValue = labelProb[0].max()
            txt.text = labelProb[0].indexOf(maxValue!!).toString()

        }
    }

    @Throws(IOException::class)
    private fun loadModelFile(assets: AssetManager, modelFilename: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelFilename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.getChannel()
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun mnistizeImage(bitmap: Bitmap): Bitmap {
        val mnistSizeBitmap = Bitmap.createScaledBitmap(bitmap, 28, 28, true)
        val resultImage = Bitmap.createBitmap(28,28,Bitmap.Config.ARGB_8888)
        val canvas = Canvas(resultImage)
        val ma = ColorMatrix()
        ma.setSaturation(0F)
        ma.set(NEGATIVE)
        val paint = Paint()
        paint.colorFilter = ColorMatrixColorFilter(ma)
        canvas.drawBitmap(mnistSizeBitmap,0f,0f,paint)
        return resultImage
    }
}
