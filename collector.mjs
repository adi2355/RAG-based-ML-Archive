#!/usr/bin/env node

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import inquirer from 'inquirer';
import chalk from 'chalk';
import { glob } from 'glob';

// Default configuration for file collection
const DEFAULT_CONFIG = {
  extensions: ['.ts', '.tsx', '.js', '.jsx', '.md', '.json', '.py', '.sql', '.html', '.css'],
  excludeDirs: [
    '.expo',
    'android',
    'ios',
    'node_modules',
    'static',
    'templates',
    'venv',
    'dist',
    'build',
    '.next',
    '.git'
  ],
  excludeFiles: [
    '.test.',
    '.spec.',
    '.d.ts',
    '.map',
    'next-env.d.ts',
    '.gitignore',
    '.eslintrc',
    'package-lock.json',
  ],
  excludePaths: [],
  includePaths: [],
  selectedFiles: [],
  maxFileSize: 1024 * 1024, // 1MB
};

// Helper function to get relative path
function getRelativePath(fullPath, rootDir) {
  return path.relative(rootDir, fullPath);
}

// Check if a file should be excluded based on config
function shouldExcludeFile(filePath, config) {
  const normalizedPath = path.normalize(filePath);
  
  // Check for explicit includes in the selectedFiles list
  if (config.selectedFiles.length > 0) {
    if (config.selectedFiles.includes(normalizedPath)) {
      return false;
    }
  }
  
  // Check for explicit includes
  if (config.includePaths.length > 0) {
    const isIncluded = config.includePaths.some(includePath => 
      normalizedPath.includes(path.normalize(includePath))
    );
    if (!isIncluded) return true;
  }

  // Check for explicit excludes
  if (config.excludePaths.some(excludePath =>
    normalizedPath.includes(path.normalize(excludePath))
  )) {
    return true;
  }

  if (config.excludeFiles.some(pattern => normalizedPath.includes(pattern))) {
    return true;
  }

  return false;
}

// Check if a directory should be excluded based on config
function shouldExcludeDir(dirPath, config) {
  const normalizedPath = path.normalize(dirPath);
  return config.excludeDirs.some(excludeDir =>
    normalizedPath.includes(path.normalize(excludeDir))
  );
}

// Find all matching files based on current config
async function findMatchingFiles(config) {
  const rootDir = process.cwd();
  
  // If we have specific selected files, just use those
  if (config.selectedFiles && config.selectedFiles.length > 0) {
    const results = [];
    for (const filePath of config.selectedFiles) {
      try {
        const fullPath = path.isAbsolute(filePath) ? filePath : path.join(rootDir, filePath);
        if (fs.existsSync(fullPath) && fs.statSync(fullPath).isFile()) {
          const relativePath = getRelativePath(fullPath, rootDir);
          results.push({ path: fullPath, relativePath });
        }
      } catch (error) {
        console.error(chalk.red(`Error processing file ${filePath}:`), error);
      }
    }
    return results;
  }
  
  // Otherwise, do normal directory scanning
  return collectFiles(rootDir, rootDir, config);
}

// Recursively collect files from directory
function collectFiles(dir, rootDir, config) {
  let results = [];
  try {
    const items = fs.readdirSync(dir, { withFileTypes: true });

    for (const item of items) {
      const fullPath = path.join(dir, item.name);
      const relativePath = getRelativePath(fullPath, rootDir);

      if (item.isDirectory()) {
        if (!shouldExcludeDir(fullPath, config)) {
          results = results.concat(collectFiles(fullPath, rootDir, config));
        }
      } else {
        const ext = path.extname(item.name).toLowerCase();

        if (config.extensions.includes(ext) && !shouldExcludeFile(relativePath, config)) {
          const stats = fs.statSync(fullPath);
          if (stats.size <= config.maxFileSize) {
            results.push({ path: fullPath, relativePath });
          }
        }
      }
    }
  } catch (error) {
    console.error(chalk.red(`Error reading directory ${dir}:`), error);
  }

  return results;
}

// Preview files that would be collected
async function previewFiles(config) {
  const files = await findMatchingFiles(config);
  
  console.log(chalk.blue('\nFiles that would be collected:'));
  console.log(chalk.yellow(`Found ${files.length} files matching criteria\n`));
  
  if (files.length > 20) {
    const { showMore } = await inquirer.prompt({
      type: 'confirm',
      name: 'showMore',
      message: 'Show all files? (20 of ' + files.length + ')',
      default: false
    });
    
    if (showMore) {
      files.forEach(file => console.log(chalk.green(file.relativePath)));
    } else {
      files.slice(0, 20).forEach(file => console.log(chalk.green(file.relativePath)));
      console.log(chalk.gray(`... and ${files.length - 20} more files`));
    }
  } else {
    files.forEach(file => console.log(chalk.green(file.relativePath)));
  }
  
  return files.length;
}

// Get directories in the current project
async function getDirectories() {
  const rootDir = process.cwd();
  try {
    const dirs = await glob('*/', { ignore: ['node_modules/**', '.git/**'] });
    return dirs.map(dir => dir.replace(/\/$/, ''));
  } catch (error) {
    console.error(chalk.red('Error getting directories:'), error);
    return [];
  }
}

// Extract the common file extensions in the project
async function getFileExtensions() {
  try {
    const allFiles = await glob('**/*.*', { 
      ignore: ['node_modules/**', '.git/**'],
      nodir: true
    });
    
    const extensions = new Set();
    allFiles.forEach(file => {
      const ext = path.extname(file).toLowerCase();
      if (ext) extensions.add(ext);
    });
    
    return Array.from(extensions);
  } catch (error) {
    console.error(chalk.red('Error getting file extensions:'), error);
    return DEFAULT_CONFIG.extensions;
  }
}

// Find all files matching patterns from a list
async function findFilesMatchingPatterns(patterns) {
  const results = [];
  const rootDir = process.cwd();
  
  for (const pattern of patterns) {
    try {
      // Split patterns to handle "dir/*.js" format
      const files = await glob(pattern, { ignore: ['node_modules/**', '.git/**'], nodir: true });
      for (const file of files) {
        const fullPath = path.join(rootDir, file);
        if (fs.existsSync(fullPath) && fs.statSync(fullPath).isFile()) {
          results.push(file);
        }
      }
    } catch (error) {
      console.error(chalk.red(`Error with pattern ${pattern}:`), error);
    }
  }
  
  return [...new Set(results)]; // Remove duplicates
}

// Save configuration for future use
function saveConfig(config, filename = 'collector-config.json') {
  try {
    fs.writeFileSync(filename, JSON.stringify(config, null, 2), 'utf8');
    console.log(chalk.green(`Configuration saved to ${filename}`));
  } catch (error) {
    console.error(chalk.red('Error saving config:'), error);
  }
}

// Load configuration
function loadConfig(filename = 'collector-config.json') {
  try {
    if (fs.existsSync(filename)) {
      const configJson = fs.readFileSync(filename, 'utf8');
      return JSON.parse(configJson);
    }
  } catch (error) {
    console.error(chalk.red('Error loading config:'), error);
  }
  return { ...DEFAULT_CONFIG };
}

// Save the output file
async function saveOutput(outputFile, config = {}) {
  const rootDir = process.cwd();
  
  // Get all matching files
  const files = await findMatchingFiles(config);
  
  if (files.length === 0) {
    console.log(chalk.yellow('\nNo files match your criteria. Please adjust settings.'));
    return;
  }
  
  try {
    console.log(chalk.blue(`\nProcessing ${files.length} files...`));
    fs.writeFileSync(outputFile, '');
    
    // Add files content
    files.forEach(({ path: filePath, relativePath }, index) => {
      try {
        const content = fs.readFileSync(filePath, 'utf8');
        const separator = '='.repeat(80) + `\nFile: ${relativePath}\n` + '='.repeat(80);
        
        fs.appendFileSync(
          outputFile,
          `\n\n${separator}\n\n${content}`
        );
        
        // Show progress
        if (index > 0 && index % 10 === 0) {
          process.stdout.write(`${chalk.gray(index)} `);
        }
      } catch (error) {
        console.error(chalk.red(`\nError processing file ${relativePath}:`), error);
      }
    });
    
    console.log(chalk.green(`\n\nOutput saved to ${outputFile}`));
    console.log(chalk.gray(`Contains ${files.length} files`));
  } catch (error) {
    console.error(chalk.red('\nError saving output:'), error);
  }
}

// Format a file list for display
function formatFileList(files, max = 5) {
  if (files.length === 0) {
    return chalk.gray('(None)');
  }
  
  if (files.length <= max) {
    return files.map(f => chalk.green(f)).join('\n  ');
  }
  
  return files.slice(0, max).map(f => chalk.green(f)).join('\n  ') + 
         `\n  ${chalk.gray(`... and ${files.length - max} more`)}`;
}

// Interactive menu for configuring and collecting code
async function interactiveCollector() {
  console.log(chalk.blue('='.repeat(50)));
  console.log(chalk.blue('Interactive Code Collector'));
  console.log(chalk.blue('='.repeat(50)));
  
  let config = { ...DEFAULT_CONFIG, selectedFiles: [] };
  let configLoaded = false;
  
  // Check if there's a saved config
  if (fs.existsSync('collector-config.json')) {
    const { loadSavedConfig } = await inquirer.prompt({
      type: 'confirm',
      name: 'loadSavedConfig',
      message: 'Found a saved configuration. Would you like to load it?',
      default: true
    });
    
    if (loadSavedConfig) {
      config = loadConfig();
      if (!config.selectedFiles) config.selectedFiles = [];
      configLoaded = true;
      console.log(chalk.green('Configuration loaded!'));
    }
  }

  let exitMenu = false;
  while (!exitMenu) {
    const fileCount = await previewFiles(config);
    
    // Display current selection summary
    console.log('\nCurrent selection:');
    if (config.selectedFiles.length > 0) {
      console.log(chalk.yellow(`${config.selectedFiles.length} specific files selected`));
    } else if (config.includePaths.length > 0) {
      console.log(`${chalk.yellow(config.includePaths.length)} include patterns:\n  ${formatFileList(config.includePaths)}`);
    }
    
    console.log(`${chalk.yellow(config.extensions.length)} file extensions:\n  ${formatFileList(config.extensions)}`);
    
    if (config.excludeDirs.length > 0) {
      console.log(`${chalk.yellow(config.excludeDirs.length)} excluded directories:\n  ${formatFileList(config.excludeDirs, 3)}`);
    }
    
    // Show total matching files
    console.log(chalk.blue(`\nTotal matching files: ${chalk.yellow(fileCount)}`));
    
    const { action } = await inquirer.prompt({
      type: 'list',
      name: 'action',
      message: 'What would you like to do?',
      choices: [
        'Collect files with current settings',
        new inquirer.Separator('─── File Selection ───'),
        'Add specific files (by name/pattern)',
        config.selectedFiles.length > 0 ? 'View/Remove selected files' : null,
        'Select file extensions',
        'Choose directories to include',
        'Choose directories to exclude',
        'Add files/patterns to exclude',
        new inquirer.Separator('─── Tools ───'),
        'Preview files that would be collected',
        'Save current configuration',
        configLoaded ? 'Reset to loaded configuration' : 'Reset to default configuration',
        'Exit'
      ].filter(Boolean)
    });
    
    switch (action) {
      case 'Collect files with current settings':
        const { outputFile } = await inquirer.prompt({
          type: 'input',
          name: 'outputFile',
          message: 'Enter output file name:',
          default: 'code-collection.txt'
        });
        
        await saveOutput(outputFile, config);
        break;
        
      case 'Add specific files (by name/pattern)':
        const { filePatterns } = await inquirer.prompt({
          type: 'input',
          name: 'filePatterns',
          message: 'Enter files or patterns (comma or space separated, e.g., *.py, downloader.py):',
        });
        
        if (filePatterns.trim()) {
          // Parse patterns, handling both comma and space separation
          const patterns = filePatterns
            .split(/[,\s]+/)
            .map(f => f.trim())
            .filter(f => f);
            
          // Find actual files matching patterns
          const matchedFiles = await findFilesMatchingPatterns(patterns);
          
          if (matchedFiles.length > 0) {
            config.selectedFiles = [...new Set([...config.selectedFiles, ...matchedFiles])];
            console.log(chalk.green(`\nAdded ${matchedFiles.length} files to selection`));
          } else {
            console.log(chalk.yellow('\nNo files matched your patterns.'));
          }
        }
        break;
        
      case 'View/Remove selected files':
        if (config.selectedFiles.length === 0) {
          console.log(chalk.yellow('No files are currently selected.'));
          break;
        }
        
        const { filesToRemove } = await inquirer.prompt({
          type: 'checkbox',
          name: 'filesToRemove',
          message: 'Select files to remove:',
          choices: config.selectedFiles,
          pageSize: 20
        });
        
        if (filesToRemove.length > 0) {
          config.selectedFiles = config.selectedFiles.filter(f => !filesToRemove.includes(f));
          console.log(chalk.green(`Removed ${filesToRemove.length} files from selection`));
        }
        break;
        
      case 'Select file extensions':
        const availableExtensions = await getFileExtensions();
        const { selectedExtensions } = await inquirer.prompt({
          type: 'checkbox',
          name: 'selectedExtensions',
          message: 'Select file extensions to include:',
          choices: availableExtensions,
          default: config.extensions,
          pageSize: 15
        });
        
        config.extensions = selectedExtensions;
        console.log(chalk.green(`Selected ${selectedExtensions.length} extensions`));
        break;
        
      case 'Choose directories to include':
        const directories = await getDirectories();
        const { selectedDirs } = await inquirer.prompt({
          type: 'checkbox',
          name: 'selectedDirs',
          message: 'Select directories to INCLUDE (leave empty for all):',
          choices: directories
        });
        
        if (selectedDirs.length > 0) {
          config.includePaths = selectedDirs.map(dir => `${dir}/`);
          console.log(chalk.green(`Selected ${selectedDirs.length} directories to include`));
        } else {
          config.includePaths = [];
          console.log(chalk.yellow('No directories specifically included (will include all non-excluded)'));
        }
        break;
        
      case 'Choose directories to exclude':
        const allDirs = await getDirectories();
        const { dirsToExclude } = await inquirer.prompt({
          type: 'checkbox',
          name: 'dirsToExclude',
          message: 'Select directories to EXCLUDE:',
          choices: allDirs,
          default: allDirs.filter(dir => config.excludeDirs.includes(dir))
        });
        
        config.excludeDirs = DEFAULT_CONFIG.excludeDirs
          .filter(dir => !allDirs.includes(dir))
          .concat(dirsToExclude);
          
        console.log(chalk.green(`Updated exclude list: ${dirsToExclude.length} directories`));
        break;
        
      case 'Add files/patterns to exclude':
        const { filesToExclude } = await inquirer.prompt({
          type: 'input',
          name: 'filesToExclude',
          message: 'Enter patterns to exclude (comma separated, e.g., .min.js,backup/):',
        });
        
        if (filesToExclude.trim()) {
          const newPatterns = filesToExclude
            .split(',')
            .map(f => f.trim())
            .filter(f => f);
            
          config.excludeFiles = [...config.excludeFiles, ...newPatterns];
          console.log(chalk.green(`Added ${newPatterns.length} exclusion patterns`));
        }
        break;
        
      case 'Preview files that would be collected':
        // Already shown at menu start
        break;
        
      case 'Save current configuration':
        const { configFilename } = await inquirer.prompt({
          type: 'input',
          name: 'configFilename',
          message: 'Save configuration to file:',
          default: 'collector-config.json'
        });
        
        saveConfig(config, configFilename);
        configLoaded = true;
        break;
        
      case 'Reset to default configuration':
      case 'Reset to loaded configuration':
        if (configLoaded) {
          config = loadConfig();
          if (!config.selectedFiles) config.selectedFiles = [];
          console.log(chalk.green('Configuration reset to saved settings'));
        } else {
          config = { ...DEFAULT_CONFIG, selectedFiles: [] };
          console.log(chalk.green('Configuration reset to defaults'));
        }
        break;
        
      case 'Exit':
        exitMenu = true;
        break;
    }
  }
  
  console.log(chalk.blue('Goodbye!'));
}

// Main entry point
async function main() {
  try {
    await interactiveCollector();
  } catch (error) {
    console.error(chalk.red('\nError:'), error);
    process.exit(1);
  }
}

// Check if running as main module
const isMainModule = process.argv[1] === fileURLToPath(import.meta.url);

if (isMainModule) {
  main();
}

export { interactiveCollector };