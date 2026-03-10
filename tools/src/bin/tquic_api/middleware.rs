//! Axum middleware for the tquic_api server.
//!
//! Currently provides a single `log_request` layer that emits an INFO log line
//! for every incoming request (method + URI) and its response status code.

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
