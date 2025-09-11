# Model Routing

> Route requests dynamically between AI models. Learn how to use OpenRouter's Auto Router and model fallback features for optimal performance and reliability.

OpenRouter provides two options for model routing.

## Auto Router

The [Auto Router](https://openrouter.ai/openrouter/auto), a special model ID that you can use to choose between selected high-quality models based on your prompt, powered by [NotDiamond](https://www.notdiamond.ai/).

```json
{
  "model": "openrouter/auto",
  ... // Other params
}
```

The resulting generation will have `model` set to the model that was used.

## The `models` parameter

The `models` parameter lets you automatically try other models if the primary model's providers are down, rate-limited, or refuse to reply due to content moderation.

```json
{
  "models": ["anthropic/claude-3.5-sonnet", "gryphe/mythomax-l2-13b"],
  ... // Other params
}
```

If the model you selected returns an error, OpenRouter will try to use the fallback model instead. If the fallback model is down or returns an error, OpenRouter will return that error.

By default, any error can trigger the use of a fallback model, including context length validation errors, moderation flags for filtered models, rate-limiting, and downtime.

Requests are priced using the model that was ultimately used, which will be returned in the `model` attribute of the response body.

## Using with OpenAI SDK

To use the `models` array with the OpenAI SDK, include it in the `extra_body` parameter. In the example below, gpt-4o will be tried first, and the `models` array will be tried in order as fallbacks.

<Template
  data={{
  API_KEY_REF,
}}
>
  <CodeGroup>
    ```typescript
    import OpenAI from 'openai';

    const openrouterClient = new OpenAI({
      baseURL: 'https://openrouter.ai/api/v1',
      // API key and headers
    });

    async function main() {
      // @ts-expect-error
      const completion = await openrouterClient.chat.completions.create({
        model: 'openai/gpt-4o',
        models: ['anthropic/claude-3.5-sonnet', 'gryphe/mythomax-l2-13b'],
        messages: [
          {
            role: 'user',
            content: 'What is the meaning of life?',
          },
        ],
      });
      console.log(completion.choices[0].message);
    }

    main();
    ```

    ```python
    from openai import OpenAI

    openai_client = OpenAI(
      base_url="https://openrouter.ai/api/v1",
      api_key={{API_KEY_REF}},
    )

    completion = openai_client.chat.completions.create(
        model="openai/gpt-4o",
        extra_body={
            "models": ["anthropic/claude-3.5-sonnet", "gryphe/mythomax-l2-13b"],
        },
        messages=[
            {
                "role": "user",
                "content": "What is the meaning of life?"
            }
        ]
    )

    print(completion.choices[0].message.content)
    ```
  </CodeGroup>
</Template>
