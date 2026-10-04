export default {
  testDir: ".",
  testMatch: /.*\.spec\.mjs/,
  timeout: 600000,
  workers: 1,
  reporter: [["list"]],
  use: { trace: "off" },
};
