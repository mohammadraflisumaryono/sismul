import os
from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
import cv2
from PIL import Image
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Konfigurasi folder output
UPLOAD_FOLDER = 'output'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
EOF_MARKER = '1111111111111110'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi Embed (Parity Modulation DCT)
def embed_message(img, message):
    try:
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        y_channel = img_yuv[:, :, 0]
        y_dct = cv2.dct(np.float32(y_channel))
        flat_dct = y_dct.flatten()

        message_bits = ''.join(format(ord(c), '08b') for c in message) + EOF_MARKER
        if len(message_bits) > len(flat_dct) - 10:
            return None, "Pesan terlalu panjang untuk gambar ini."

        embed_idx = 10
        for bit in message_bits:
            while embed_idx < len(flat_dct):
                val = flat_dct[embed_idx]
                if abs(val) < 2.0:  # Ambang batas lebih tinggi
                    embed_idx += 1
                    continue
                parity_val = int(np.floor(abs(val))) % 2
                if bit == '1' and parity_val == 0:
                    flat_dct[embed_idx] += 1 if val >= 0 else -1
                elif bit == '0' and parity_val == 1:
                    flat_dct[embed_idx] += 1 if val >= 0 else -1
                embed_idx += 1
                break

        y_dct_modified = flat_dct.reshape(y_channel.shape)
        y_idct = cv2.idct(y_dct_modified)
        img_yuv[:, :, 0] = np.clip(y_idct, 0, 255)
        encoded_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        return encoded_img.astype(np.uint8), "Pesan berhasil disisipkan."

    except Exception as e:
        return None, f"Error saat menyisipkan pesan: {str(e)}"

# Fungsi Ekstraksi
def extract_message(img, img_path=None):
    try:
        # Validasi format PNG
        if img_path:
            img_pil = Image.open(img_path)
            if img_pil.format != 'PNG':
                return "Gambar harus dalam format PNG untuk ekstraksi."

        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        y_channel = img_yuv[:, :, 0]
        y_dct = cv2.dct(np.float32(y_channel))
        flat_dct = y_dct.flatten()

        bits = ""
        max_bits = 10000  # Batas maksimum untuk mencegah loop tak terbatas
        for i in range(10, min(len(flat_dct), 10 + max_bits)):
            val = flat_dct[i]
            if abs(val) < 2.0:
                continue
            parity = int(np.floor(abs(val))) % 2
            bits += str(parity)
            if bits.endswith(EOF_MARKER):
                bits = bits[:-len(EOF_MARKER)]
                break
        else:
            return "Tidak ditemukan pesan valid."

        if len(bits) % 8 != 0:
            bits = bits[:-(len(bits) % 8)]

        chars = []
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            try:
                chars.append(chr(int(byte, 2)))
            except ValueError:
                continue
        return ''.join(chars) if chars else "Pesan tidak dapat dibaca."

    except Exception as e:
        return f"Error saat ekstraksi: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/embed', methods=['POST'])
def embed():
    if 'file' not in request.files or not request.form.get('message'):
        return jsonify({'error': 'Gambar atau pesan tidak diberikan.'}), 400
    file = request.files['file']
    message = request.form.get('message')

    if not allowed_file(file.filename):
        return jsonify({'error': 'Format file tidak didukung. Gunakan PNG/JPG/JPEG.'}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    img = cv2.imread(input_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encoded_img, message_result = embed_message(img_rgb, message)

    if encoded_img is None:
        return jsonify({'error': message_result}), 500

    output_filename = f"stego_{filename.split('.')[0]}.png"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    Image.fromarray(encoded_img).save(output_path, format='PNG', quality=100, compress_level=0)

    return jsonify({'message': message_result, 'file': f'/download/{output_filename}'})

@app.route('/extract', methods=['POST'])
def extract():
    if 'file' not in request.files:
        return jsonify({'error': 'Gambar tidak diberikan.'}), 400
    file = request.files['file']

    if not allowed_file(file.filename):
        return jsonify({'error': 'Format file tidak didukung. Gunakan PNG.'}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    img = cv2.imread(input_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    extracted_message = extract_message(img_rgb, input_path)

    return jsonify({'message': extracted_message})

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)