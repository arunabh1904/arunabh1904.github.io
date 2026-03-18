import { execFileSync } from "node:child_process";
import process from "node:process";

if (process.env.CI) {
  process.exit(0);
}

try {
  execFileSync("git", ["rev-parse", "--is-inside-work-tree"], {
    stdio: "ignore",
  });
} catch {
  process.exit(0);
}

try {
  execFileSync("git", ["config", "--local", "core.hooksPath", ".githooks"], {
    stdio: "ignore",
  });
  process.stdout.write("Configured git hooks at .githooks\n");
} catch (error) {
  process.stderr.write(`Failed to configure git hooks: ${error.message}\n`);
  process.exit(1);
}
