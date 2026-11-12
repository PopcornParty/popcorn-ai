
export default async function handler(req, res) {
  res.setHeader("Content-Type", "application/json");

  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Use POST." });
  }

  const apiKey = process.env.OPENROUTER_API_KEY;

  if (!apiKey) {
    return res.status(500).json({
      error: "OPENROUTER_API_KEY is missing from Vercel."
    });
  }

  try {
    const body = typeof req.body === "string"
      ? JSON.parse(req.body)
      : req.body;

    if (!Array.isArray(body?.messages) || !body.messages.length) {
      return res.status(400).json({
        error: "No messages provided."
      });
    }

    const messages = body.messages.slice(-30).map(m => {
      if (!["user", "assistant"].includes(m.role) ||
          typeof m.content !== "string") {
        throw new Error("Invalid message format.");
      }

      return {
        role: m.role,
        content: m.content.slice(0, 12000)
      };
    });

    const response = await fetch(
      "https://openrouter.ai/api/v1/chat/completions",
      {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${apiKey}`,
          "Content-Type": "application/json",
          "X-Title": "Popcorn AI"
        },
        body: JSON.stringify({
          model: "openrouter/free",
          messages,
          temperature: 0.7
        })
      }
    );

    const raw = await response.text();

    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      return res.status(502).json({
        error: "AI provider returned an invalid response."
      });
    }

    if (!response.ok) {
      return res.status(502).json({
        error: data.error?.message || "AI provider request failed."
      });
    }

    const reply = data.choices?.[0]?.message?.content;

    if (!reply) {
      return res.status(502).json({
        error: "AI returned an empty response."
      });
    }

    return res.status(200).json({ reply });

  } catch (error) {
    console.error(error);

    return res.status(500).json({
      error: error.message || "Server error."
    });
  }
}
