use axum::extract::Request;
use axum::middleware::Next;
use axum::response::Response;

/// Log every incoming request and its response status at INFO level.
pub async fn log_request(req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let uri    = req.uri().clone();
    log::info!("→  {method} {uri}");
    let resp = next.run(req).await;
    log::info!("←  {method} {uri}  {}", resp.status());
    resp
}
