from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import asyncio

from vector_index import add_to_index, search_index
from fast_segmentation import segment_objects, draw_boxes_and_metadata

app = FastAPI()

@app.post("/upload_video/")
async def upload_video(file: UploadFile = File(...)):
    video_stream = cv2.VideoCapture(file.file)
    frame_queue = asyncio.Queue()

    async def process_video():
        while video_stream.isOpened():
            ret, frame = video_stream.read()
            if not ret:
                break
            segmented_objects = segment_objects(frame)
            objects_with_metadata = []
            for obj, bbox in segmented_objects:
                results = search_index(obj)
                if results:
                    metadata = results[0]  # Using the first result as an example
                    objects_with_metadata.append((obj, bbox, metadata))
            frame_with_metadata = draw_boxes_and_metadata(frame, objects_with_metadata)
            _, buffer = cv2.imencode('.jpg', frame_with_metadata)
            frame_bytes = buffer.tobytes()
            await frame_queue.put(frame_bytes)
        video_stream.release()
        await frame_queue.put(None)  # Signal that the stream is finished

    asyncio.create_task(process_video())

    async def video_streamer():
        while True:
            frame = await frame_queue.get()
            if frame is None:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return StreamingResponse(video_streamer(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.post("/add_image/")
async def add_image(file: UploadFile = File(...), metadata: dict = None):
    image = Image.open(BytesIO(await file.read()))
    add_to_index(image, metadata)
    return {"message": "Image added"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
