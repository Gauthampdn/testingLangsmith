import * as dotenv from "dotenv";
dotenv.config();

import readline from "readline";
import { z } from "zod";

import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
// We'll wrap the Gemini model with LangSmith tracing:
import { wrapSDK } from "langsmith/wrappers"; 

import OpenAI from "openai";

import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { createToolCallingAgent, AgentExecutor } from "langchain/agents";
import { DynamicStructuredTool } from "@langchain/core/tools";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { dirname } from "path";
// Import play-sound with ES modules syntax
import pkg from "play-sound";
const player = pkg({});

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Initialize the Gemini model and then wrap it with LangSmith for global tracing
const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash-lite",
  maxOutputTokens: 2048,
  streaming: true,
});
const wrappedModel = wrapSDK(model);

// Your grocery list object
const groceryList = {
  fruits: [],
  vegetables: [],
};

// Schema definitions
const CategorySchema = z.enum(["fruits", "vegetables"]);
const ItemSchema = z.string().min(1).toLowerCase();

// Initialize OpenAI client for audio (if you need it)
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY, // Ensure your API key is set in the environment variables
});

// Define the tools using DynamicStructuredTool
const addToListTool = new DynamicStructuredTool({
  name: "add_to_list",
  description: "Add items to the grocery list",
  schema: z.object({
    category: z.enum(["fruits", "vegetables"]),
    item: z.string().min(1),
  }),
  func: async ({ category, item }) => {
    // Process asynchronously
    const promise = new Promise((resolve) => {
      setTimeout(() => {
        try {
          const normalizedItem = item.toLowerCase();
          if (category === "fruits") {
            groceryList.fruits.push(normalizedItem);
            resolve(`Added ${normalizedItem} to fruits list`);
          } else if (category === "vegetables") {
            groceryList.vegetables.push(normalizedItem);
            resolve(`Added ${normalizedItem} to vegetables list`);
          }
        } catch (error) {
          resolve("Error adding item to list");
        }
      }, 0);
    });
    return `Started adding ${item} to ${category} list...`;
  },
});

const retrieveListTool = new DynamicStructuredTool({
  name: "retrieve_list",
  description: "Retrieve all items from the grocery list",
  schema: z.object({
    // Dummy field
    dummy: z.string().optional(),
  }),
  func: async () => {
    return JSON.stringify({
      fruits: [...groceryList.fruits],
      vegetables: [...groceryList.vegetables],
      status: "Some items may still be processing...",
    });
  },
});

// Update tools array
const tools = [addToListTool, retrieveListTool];

const prompt = ChatPromptTemplate.fromMessages([
  [
    "system",
    `You are a helpful assistant that manages a grocery list and explains topics.

When users want to add items:
1. Determine if the item is a fruit or vegetable
2. Use the add_to_list tool with the correct category
3. Confirm what was added

When users ask to see the list:
1. Use the retrieve_list tool
2. Format the response in a readable way

Always be friendly and helpful in your responses.`,
  ],
  new MessagesPlaceholder("chat_history"),
  ["human", "{input}"],
  new MessagesPlaceholder("agent_scratchpad"),
]);

// Create the agent using the wrapped model
const agent = await createToolCallingAgent({
  llm: model, // use the wrapped model so all calls are auto-traced
  prompt,
  tools,
});

// Create the executor with a max iteration and config (tracing is now automatic via the wrapped model)
const agentExecutor = new AgentExecutor({
  agent,
  tools,
  maxIterations: 10,
}).withConfig({
  tags: ["grocery"],
  metadata: {
    app_version: "1.0.0",
    environment: "development",
  },
});

// Initialize readline interface
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const chat_history = [];

// Function to convert text to speech and play it
async function textToSpeech(text) {
  try {
    const speechFile = path.resolve(__dirname, "./speech.mp3");

    const mp3 = await openai.audio.speech.create({
      model: "tts-1-hd",
      voice: "alloy",
      input: text,
    });

    const buffer = Buffer.from(await mp3.arrayBuffer());
    await fs.promises.writeFile(speechFile, buffer);

    // Play the audio file
    return new Promise((resolve, reject) => {
      player.play(speechFile, (err) => {
        if (err) {
          console.error("Error playing audio:", err);
          reject(err);
        }
        resolve();
      });
    });
  } catch (error) {
    console.error("Error generating or playing speech:", error);
  }
}

function askQuestion() {
  rl.question("User: ", async (input) => {
    if (input.toLowerCase() === "exit") {
      rl.close();
      return;
    }

    try {
      const response = await agentExecutor.invoke({
        input: input,
        chat_history: chat_history,
      });

      console.log("Assistant: ", response.output);

      // Convert assistant's response to speech and wait for it to finish playing
      await textToSpeech(response.output);

      chat_history.push(new HumanMessage(input));
      chat_history.push(new AIMessage(response.output));

      askQuestion();
    } catch (error) {
      console.error("Error:", error);
      askQuestion();
    }
  });
}

askQuestion();
