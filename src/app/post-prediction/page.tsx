"use client";

import React, { useState } from "react";
import { Paper, Title, Stack, TextInput, Button, Select, Textarea, Loader, Text, Box, Grid, ScrollArea, Notification } from "@mantine/core";

// Function to format AI response (clickable links, bullet points, bold text)
const formatLlmResponse = (response: string) => {
  let formattedResponse = response.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  formattedResponse = formattedResponse.replace(
    /(https?:\/\/[^\s]+)/g,
    '<a href="$1" target="_blank" rel="noopener noreferrer" class="text-blue-400 underline">$1</a>'
  );
  formattedResponse = formattedResponse.replace(/- /g, "<br>• ");
  return formattedResponse;
};

// Function to color-code prediction results
const getPredictionColor = (prediction: string) => {
  switch (prediction.toLowerCase()) {
    case "popular":
      return "green";
    case "decent":
      return "orange";
    case "unpopular":
      return "red";
    default:
      return "gray";
  }
};

export default function PostPredictionPage() {
  const [textInput, setTextInput] = useState("");
  const [postTitle, setPostTitle] = useState("");
  const [category, setCategory] = useState("A Level");
  const [llmResponse, setLlmResponse] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<Record<string, any> | null>(null);
  const [analysisHistory, setAnalysisHistory] = useState<Record<string, any>[]>([]);
  const [loading, setLoading] = useState(false);
  const [alertMessage, setAlertMessage] = useState<string | null>(null);

  const handlePrediction = async (event: React.FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setAlertMessage(null);

    try {
      const response = await fetch("https://validate-post-78306345447.asia-southeast1.run.app/validate_post", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: textInput, title: postTitle, category }),
      });

      if (response.ok) {
        const data = await response.json();
        const { ridiculous, leaks_pii, relevant_to_category } = data.result;

        if (ridiculous || leaks_pii || !relevant_to_category) {
          let issues = [];
          if (ridiculous) issues.push("Ridiculous content");
          if (leaks_pii) issues.push("Contains PII (personal information)");
          if (!relevant_to_category) issues.push("Not relevant to the selected category");

          setAlertMessage(`Post issue detected: ${issues.join(", ")}`);
          setAnalysisResult(null);
          setLlmResponse(data.response);
          return;
        }

        const predictionResponse = await fetch("https://post-validation-78306345447.asia-southeast1.run.app/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ post_content: textInput, post_title: postTitle, category }),
        });

        if (predictionResponse.ok) {
          const predictionData = await predictionResponse.json();
          setAnalysisResult(predictionData);

          // Store analysis in history
          setAnalysisHistory((prevHistory) => [
            { category, title: postTitle, content: textInput, predictions: predictionData.predictions },
            ...prevHistory,
          ]);
        } else {
          const error = await predictionResponse.json();
          setLlmResponse(`Error with prediction: ${error.error}`);
        }

        setLlmResponse(data.response);
      } else {
        const error = await response.json();
        setLlmResponse(`Error: ${error.error}`);
      }
    } catch (error) {
      setLlmResponse("Error connecting to the server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper shadow="xs" p="xl" radius="md" style={{ minHeight: "100vh", backgroundColor: "#1A1B1E" }}>
      <Title order={2} mb="lg" c="white" style={{ textAlign: "center" }}>
        Post Prediction & Analysis
      </Title>

      <Grid gutter="lg">
        {/* Left Side (Form + AI Feedback) */}
        <Grid.Col span={{ base: 12, md: 8 }}>
          <Stack>
            {/* Form Section */}
            <Paper p="lg" radius="md" shadow="md" style={{ backgroundColor: "#2C2E33" }}>
              <Title order={3} mb="md" c="white">
                Enter Your Post Details
              </Title>
              {alertMessage && (
                <Notification color="red" onClose={() => setAlertMessage(null)}>
                  {alertMessage}
                </Notification>
              )}
              <form onSubmit={handlePrediction}>
                <Stack>
                  {/* Category Select */}
                  <Select
                    label="Select Category"
                    value={category}
                    onChange={(value) => setCategory(value || "A Level")}
                    data={["A Level", "GCSE", "Job Experience", "Study Support"]}
                    disabled={loading}
                    radius="md"
                  />

                  {/* Post Title */}
                  <TextInput
                    label="Post Title"
                    placeholder="Enter your post title"
                    value={postTitle}
                    onChange={(e) => setPostTitle(e.target.value)}
                    disabled={loading}
                    radius="md"
                  />

                  {/* Post Content */}
                  <Textarea
                    label="Post Content"
                    placeholder="Enter your post content"
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    minRows={4}
                    disabled={loading}
                    radius="md"
                  />

                  {/* Submit Button */}
                  <Button type="submit" fullWidth color="blue" radius="md" disabled={loading}>
                    {loading ? <Loader size="sm" color="white" /> : "Validate Post"}
                  </Button>
                </Stack>
              </form>
            </Paper>

            {/* AI Feedback Section */}
            <Paper p="lg" radius="md" shadow="md" style={{ backgroundColor: "#2C2E33" }}>
              <Title order={3} mb="md" c="white">
                AI Generated Feedback
              </Title>
              
              {loading ? (
                <Stack align="center">
                  <Loader size="md" color="blue" />
                  <Text size="sm" c="gray.5">
                    Analyzing your post...
                  </Text>
                </Stack>
              ) : llmResponse ? (
                <Box
                  style={{
                    backgroundColor: "#373A40",
                    padding: "16px",
                    borderRadius: "8px",
                    color: "white",
                    fontSize: "14px",
                    lineHeight: "1.6",
                  }}
                  dangerouslySetInnerHTML={{
                    __html: formatLlmResponse(llmResponse),
                  }}
                />
              ) : (
                <Text size="sm" c="gray.5">
                  No response yet. Please try submitting your post.
                </Text>
              )}
            </Paper>
          </Stack>
        </Grid.Col>

        {/* Right Side (Analysis Result + History) */}
        <Grid.Col span={{ base: 12, md: 4 }}>
        <Paper p="lg" radius="md" shadow="md" style={{ backgroundColor: "#2C2E33" }}>
            <Title order={3} mb="md" c="white">
              Analysis Result
            </Title>

            {loading ? (
              <Stack align="center">
                <Loader size="md" color="blue" />
                <Text size="sm" c="gray.5">
                  Predicting post popularity...
                </Text>
              </Stack>
            ) : analysisResult ? (
              <Stack>
                {analysisResult.predictions.map((prediction: any, index: number) => (
                  <Text
                    key={index}
                    size="lg"
                    style={{ color: getPredictionColor(prediction.predicted_class) }}
                  >
                    Day {prediction.days_since_post}: {prediction.predicted_class}
                  </Text>
                ))}
              </Stack>
            ) : (
              <Text size="sm" c="gray.5">
                No predictions yet.
              </Text>
            )}
          </Paper>
          {/* Past Analyses (Inside a Scrollable Box) */}
          <Paper p="lg" radius="md" shadow="md" style={{ backgroundColor: "#2C2E33", marginTop: "20px" }}>
            <Title order={4} mb="md" c="white">
              Past Analyses
            </Title>
            
            <ScrollArea h={300}>
              <Stack>
                {analysisHistory.length > 0 ? (
                  analysisHistory.map((item, idx) => (
                    <Paper key={idx} p="sm" radius="md" style={{ backgroundColor: "#373A40" }}>
                      {/* Category & Title */}
                      <Text size="sm" c="gray.3">
                        {item.category}: {item.title}
                      </Text>
                      {/* Shortened Post Content */}
                      <Text size="xs" c="gray.5">
                        {item.content.slice(0, 60)}...
                      </Text>
                      {/* Prediction Results */}
                      <Stack mt="sm">
                        {item.predictions.map((prediction: any, index: number) => (
                          <Text
                            key={index}
                            size="sm"
                            style={{ color: getPredictionColor(prediction.predicted_class) }}
                          >
                            Day {prediction.days_since_post}: {prediction.predicted_class}
                          </Text>
                        ))}
                      </Stack>
                    </Paper>
                  ))
                ) : (
                  <Text size="sm" c="gray.5">
                    No past analyses yet.
                  </Text>
                )}
              </Stack>
            </ScrollArea>
          </Paper>
        </Grid.Col>
      </Grid>
    </Paper>
  );
}
