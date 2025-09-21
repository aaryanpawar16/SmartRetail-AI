from google.cloud import bigquery
import os, google.auth

def main():
    try:
        print("ENV PROJECT_ID:", os.environ.get("PROJECT_ID"))
        print("ENV GOOGLE_APPLICATION_CREDENTIALS:", os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
        creds, aff_proj = google.auth.default()
        print("google.auth.default() affiliated project:", aff_proj)
        print("Credentials type:", type(creds))
        client = bigquery.Client()
        print("client.project:", client.project)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
