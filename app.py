import os
import logging
from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
import cv2
from PIL import Image
from werkzeug.utils import secure_filename

# Konfigurasi logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction.log'),
        logging.StreamHandler()
    ]
)

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
        logging.debug("Memulai proses embedding...")
        img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        y_channel = img_yuv[:, :, 0]
        logging.debug(f"Ukuran kanal Y: {y_channel.shape}")
        if y_channel.shape[0] * y_channel.shape[1] < 4096:  # Minimal 64x64
            logging.error("Resolusi gambar terlalu kecil untuk embedding.")
            return None, "Resolusi gambar terlalu kecil (minimal 64x64)."

        y_dct = cv2.dct(np.float32(y_channel))
        flat_dct = y_dct.flatten()
        logging.debug(f"Jumlah koefisien DCT: {len(flat_dct)}")

        message_bits = ''.join(format(ord(c), '08b') for c in message) + EOF_MARKER
        logging.debug(f"Panjang bit pesan (termasuk EOF): {len(message_bits)}")
        logging.debug(f"Bit pesan: {message_bits}")
        if len(message_bits) > len(flat_dct) - 10:
            logging.error("Pesan terlalu panjang untuk jumlah koefisien DCT.")
            return None, "Pesan terlalu panjang untuk gambar ini."

        valid_coeffs = sum(1 for val in flat_dct[10:] if abs(val) >= 15.0)
        logging.debug(f"Jumlah koefisien valid (>=15.0): {valid_coeffs}")
        if valid_coeffs < len(message_bits):
            logging.error("Koefisien DCT valid tidak cukup untuk pesan.")
            return None, "Koefisien DCT tidak cukup untuk pesan ini."

        embed_idx = 10
        coeff_values = []
        embed_indices = []
        for i, bit in enumerate(message_bits):
            while embed_idx < len(flat_dct):
                val = flat_dct[embed_idx]
                if abs(val) < 15.0:  # Stricter threshold
                    embed_idx += 1
                    continue
                parity_val = int(np.floor(abs(val))) % 2
                coeff_values.append((embed_idx, val, parity_val))
                embed_indices.append(embed_idx)
                logging.debug(f"Embed bit {bit} pada indeks {embed_idx}: val={val:.2f}, parity={parity_val}")
                if bit == '1' and parity_val == 0:
                    flat_dct[embed_idx] += 1 if val >= 0 else -1
                elif bit == '0' and parity_val == 1:
                    flat_dct[embed_idx] += 1 if val >= 0 else -1
                embed_idx += 1
                break
            else:
                logging.error("Koefisien DCT habis sebelum semua bit tersisip.")
                return None, "Gagal menyisipkan pesan: koefisien DCT tidak cukup."
        logging.debug(f"Koefisien yang digunakan: {coeff_values}")
        logging.debug(f"Sample modified coefficients: {[(i, flat_dct[i]) for i in embed_indices[:5]]}")

        # Save embedding indices and coefficients
        np.save(os.path.join(app.config['UPLOAD_FOLDER'], 'embed_indices.npy'), embed_indices)
        np.save(os.path.join(app.config['UPLOAD_FOLDER'], 'dct_coeffs.npy'), flat_dct)
        logging.debug("Saved embedding indices and DCT coefficients for debugging.")

        y_dct_modified = flat_dct.reshape(y_channel.shape)
        y_idct = cv2.idct(y_dct_modified)
        img_yuv[:, :, 0] = np.clip(y_idct, 0, 255)
        encoded_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        logging.info("Embedding berhasil.")
        return encoded_img.astype(np.uint8), "Pesan berhasil disisipkan."

    except Exception as e:
        logging.error(f"Error saat embedding: {str(e)}")
        return None, f"Error saat menyisipkan pesan: {str(e)}"

# Fungsi Ekstraksi
def extract_message(img, img_path=None):
    try:
        logging.debug("Memulai proses ekstraksi...")
        # Validasi format PNG
        if img_path:
            img_pil = Image.open(img_path)
            if img_pil.format != 'PNG':
                logging.error("Format file bukan PNG.")
                return "Gambar harus dalam format PNG untuk ekstraksi."

        # Load saved DCT coefficients if available
        coeff_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dct_coeffs.npy')
        if os.path.exists(coeff_path):
            flat_dct = np.load(coeff_path)
            logging.debug("Loaded saved DCT coefficients for extraction.")
        else:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            y_channel = img_yuv[:, :, 0]
            logging.debug(f"Ukuran kanal Y: {y_channel.shape}")
            if y_channel.shape[0] * y_channel.shape[1] < 4096:  # Minimal 64x64
                logging.error("Resolusi gambar terlalu kecil untuk ekstraksi.")
                return "Resolusi gambar terlalu kecil (minimal 64x64)."
            y_dct = cv2.dct(np.float32(y_channel))
            flat_dct = y_dct.flatten()
            logging.debug(f"Jumlah koefisien DCT: {len(flat_dct)}")

        logging.debug(f"Sample extracted coefficients: {[(i, flat_dct[i]) for i in range(10, 15)]}")

        # Load embedding indices if available
        embed_indices_path = os.path.join(app.config['UPLOAD_FOLDER'], 'embed_indices.npy')
        embed_indices = np.load(embed_indices_path) if os.path.exists(embed_indices_path) else None

        bits = ""
        valid_coeffs = 0
        coeff_values = []
        max_bits = 10000  # Batas maksimum untuk mencegah loop tak terbatas
        if embed_indices is not None:
            logging.debug("Using saved embedding indices for extraction.")
            for i in embed_indices:
                if i >= len(flat_dct):
                    logging.error(f"Indeks {i} melebihi panjang koefisien DCT.")
                    return "Indeks embedding tidak valid."
                val = flat_dct[i]
                if abs(val) < 15.0:  # Same threshold
                    logging.warning(f"Coefficient at index {i} below threshold: {val:.2f}")
                    continue
                valid_coeffs += 1
                parity = int(np.floor(abs(val))) % 2
                bits += str(parity)
                coeff_values.append((i, val, parity))
                logging.debug(f"Indeks {i}: val={val:.2f}, parity={parity}, bits terakhir={bits[-10:]}")
                if bits.endswith(EOF_MARKER):
                    bits = bits[:-len(EOF_MARKER)]
                    logging.info(f"EOF marker ditemukan pada indeks {i}, panjang bit: {len(bits)}")
                    break
            else:
                logging.error(f"EOF marker tidak ditemukan setelah {valid_coeffs} koefisien valid.")
                return "Gagal mengekstrak: EOF marker tidak ditemukan."
        else:
            logging.debug("No embedding indices found, using sequential extraction.")
            for i in range(10, min(len(flat_dct), 10 + max_bits)):
                val = flat_dct[i]
                if abs(val) < 15.0:
                    continue
                valid_coeffs += 1
                parity = int(np.floor(abs(val))) % 2
                bits += str(parity)
                coeff_values.append((i, val, parity))
                logging.debug(f"Indeks {i}: val={val:.2f}, parity={parity}, bits terakhir={bits[-10:]}")
                if bits.endswith(EOF_MARKER):
                    bits = bits[:-len(EOF_MARKER)]
                    logging.info(f"EOF marker ditemukan pada indeks {i}, panjang bit: {len(bits)}")
                    break
            else:
                logging.error(f"EOF marker tidak ditemukan setelah {valid_coeffs} koefisien valid.")
                return "Gagal mengekstrak: EOF marker tidak ditemukan."

        if not bits:
            logging.error("Tidak ada bit yang diekstrak.")
            return "Tidak ada pesan yang diekstrak."

        chars = []
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            if len(byte) != 8:
                logging.warning(f"Byte tidak lengkap: {byte}")
                break
            try:
                char = chr(int(byte, 2))
                if 32 <= ord(char) <= 126:  # Printable ASCII only
                    chars.append(char)
                else:
                    logging.warning(f"Karakter tidak valid pada byte {byte}: ord={ord(char)}")
                    break  # Stop at first invalid character
            except ValueError as e:
                logging.error(f"Error mengonversi byte {byte}: {str(e)}")
                break
        result = ''.join(chars) if chars else "Pesan tidak dapat dibaca."
        logging.info(f"Pesan yang diekstrak: {result}")
        return result

    except Exception as e:
        logging.error(f"Error saat ekstraksi: {str(e)}")
        return f"Error saat ekstraksi: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/embed', methods=['POST'])
def embed():
    logging.debug("Menerima permintaan embed.")
    if 'file' not in request.files or not request.form.get('message'):
        logging.error("Gambar atau pesan tidak diberikan.")
        return jsonify({'error': 'Gambar atau pesan tidak diberikan.'}), 400
    file = request.files['file']
    message = request.form.get('message')

    if not allowed_file(file.filename):
        logging.error(f"Format file tidak didukung: {file.filename}")
        return jsonify({'error': 'Format file tidak didukung. Gunakan PNG/JPG/JPEG.'}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)
    logging.debug(f"Gambar disimpan di: {input_path}")

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encoded_img, message_result = embed_message(img_rgb, message)

    if encoded_img is None:
        logging.error(f"Gagal embedding: {message_result}")
        return jsonify({'error': message_result}), 500

    output_filename = f"stego_{filename.split('.')[0]}.png"
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
    cv2.imwrite(output_path, cv2.cvtColor(encoded_img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    logging.info(f"Gambar stego disimpan di: {output_path}")

    return jsonify({'message': message_result, 'file': f'/download/{output_filename}'})

@app.route('/extract', methods=['POST'])
def extract():
    logging.debug("Menerima permintaan ekstraksi.")
    if 'file' not in request.files:
        logging.error("Gambar tidak diberikan.")
        return jsonify({'error': 'Gambar tidak diberikan.'}), 400
    file = request.files['file']

    if not allowed_file(file.filename):
        logging.error(f"Format file tidak didukung: {file.filename}")
        return jsonify({'error': 'Format file tidak didukung. Gunakan PNG.'}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)
    logging.debug(f"Gambar stego disimpan di: {input_path}")

    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    extracted_message = extract_message(img_rgb, input_path)
    logging.info(f"Hasil ekstraksi: {extracted_message}")

    return jsonify({'message': extracted_message})

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    logging.debug(f"Mengunduh file: {file_path}")
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)