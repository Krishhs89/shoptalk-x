# ShopTalk-X Streamlit frontend image. Talks to the API service over HTTP --
# no model weights in this image at all, so it stays small and rebuilds fast.
FROM python:3.11-slim

WORKDIR /app
# UI only needs a small slice of the full requirements.txt (streamlit + requests + pandas)
RUN pip install --no-cache-dir streamlit==1.40.2 requests==2.32.3 pandas==2.2.3 pyarrow==17.0.0 PyYAML==6.0.2

COPY configs/ configs/
COPY src/ src/
ENV PYTHONPATH=/app/src

EXPOSE 8501
CMD ["streamlit", "run", "src/shoptalk/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
