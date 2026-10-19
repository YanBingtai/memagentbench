use super::*;

#[tokio::test]
async fn web_search_sends_bounded_query_and_formats_results() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/"))
        .and(header("X-Subscription-Token", "test-key"))
        .and(query_param("q", "rust"))
        .and(query_param("count", "1"))
        .and(query_param("country", "CN"))
        .and(query_param("search_lang", "zh-cn"))
        .and(query_param("freshness", "pm"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "web": {
                "results": [{
                    "title": "Rust",
                    "url": "https://www.rust-lang.org/",
                    "description": "A language empowering everyone."
                }]
            }
        })))
        .expect(1)
        .mount(&server)
        .await;

    let tool =
        WebSearchTool::new("test-key", Duration::from_secs(5), 1024).with_endpoint(server.uri());
    let result = tool
        .execute_result(json!({
            "query": "rust",
            "count": 1,
            "country": "cn",
            "search_lang": "zh-cn",
            "freshness": "pm"
        }))
        .await
        .expect("web search should succeed");

    assert!(result.content.contains("Rust"));
    assert!(result.content.contains("https://www.rust-lang.org/"));
    assert_eq!(result.details["provider"], "brave");
    assert_eq!(result.details["result_count"], 1);
}

#[tokio::test]
async fn web_search_reports_provider_errors_and_empty_results() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/error"))
        .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
        .mount(&server)
        .await;
    Mock::given(method("GET"))
        .and(path("/empty"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"web": {}})))
        .mount(&server)
        .await;

    let tool = WebSearchTool::new("test-key", Duration::from_secs(5), 1024);
    let error = tool
        .clone()
        .with_endpoint(format!("{}/error", server.uri()))
        .execute_result(json!({"query": "rust"}))
        .await
        .expect_err("provider error should be returned");
    assert!(
        error
            .to_string()
            .contains("HTTP 429 Too Many Requests: rate limited"),
        "unexpected provider error: {error}"
    );

    let result = tool
        .with_endpoint(format!("{}/empty", server.uri()))
        .execute_result(json!({"query": "rust"}))
        .await
        .expect("empty provider response should be valid");
    assert_eq!(result.content, "(no results)");
    assert_eq!(result.details["result_count"], 0);
}

#[tokio::test]
async fn web_search_rejects_oversized_provider_response() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(200).set_body_bytes(vec![b'x'; MAX_WEB_RESPONSE_BYTES + 1]),
        )
        .mount(&server)
        .await;

    let error = WebSearchTool::new("test-key", Duration::from_secs(5), 1024)
        .with_endpoint(server.uri())
        .execute_result(json!({"query": "rust"}))
        .await
        .expect_err("oversized response should be rejected");
    assert!(error
        .to_string()
        .contains("search response exceeds the 2097152-byte limit"));
}

#[tokio::test]
async fn web_search_honors_cancellation_while_waiting_for_provider() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(5))
                .set_body_json(json!({"web": {"results": []}})),
        )
        .mount(&server)
        .await;

    let tool =
        WebSearchTool::new("test-key", Duration::from_secs(10), 1024).with_endpoint(server.uri());
    let cancellation = CancellationToken::new();
    let context = ToolExecutionContext::new(cancellation.clone());
    let task = tokio::spawn(async move {
        tool.execute_result_with_context(json!({"query": "rust"}), context)
            .await
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    cancellation.cancel();

    let error = task
        .await
        .expect("search task should finish")
        .expect_err("cancelled search should fail");
    assert_eq!(
        error.to_string(),
        "web search failed: search cancelled by the caller"
    );
}

#[test]
fn web_search_rejects_invalid_arguments() {
    let tool = WebSearchTool::new("test-key", Duration::from_secs(5), 1024);
    let error = tool
        .validate_arguments(&json!({"query": "", "count": 1}))
        .expect_err("empty query should fail validation");
    assert!(matches!(
        error,
        ToolError::InvalidArgumentValue { name, .. } if name == "web_search"
    ));
}
