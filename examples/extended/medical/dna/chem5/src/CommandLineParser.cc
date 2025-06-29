//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// This example is provided by the Geant4-DNA collaboration
// Any report or published results obtained using the Geant4-DNA software
// shall cite the following Geant4-DNA collaboration publication:
// Med. Phys. 37 (2010) 4692-4708
// J. Comput. Phys. 274 (2014) 841-882
// The Geant4-DNA web site is available at http://geant4-dna.org
//
// Author: Mathieu Karamitros
//
//
/// \file CommandLineParser.cc
/// \brief Implementation of the CommandLineParser class

#include "CommandLineParser.hh"

#include <iomanip>

using namespace std;
using namespace G4DNAPARSER;

CommandLineParser* CommandLineParser::fpInstance(nullptr);
G4String Command::fNoOption = "NoOption";

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

inline bool MATCH(const char* a, const char* b)
{
  return strcmp(a, b) == 0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

CommandLineParser::CommandLineParser()
{
  fpInstance = this;
  AddCommand("--help", Command::WithoutOption, "Print this help");
  AddCommand("-h", Command::WithoutOption, "Print this help");
  AddCommand("&", Command::WithoutOption);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

CommandLineParser* CommandLineParser::GetParser()
{
  if (!fpInstance) new CommandLineParser;
  return fpInstance;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

CommandLineParser::~CommandLineParser()
{
  auto it = fCommandMap.begin();
  for (; it != fCommandMap.end(); it++) {
    delete it->second;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void CommandLineParser::DeleteInstance()
{
  if (fpInstance) {
    delete fpInstance;
    fpInstance = nullptr;
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

Command::Command(Command::Type commandType, const G4String& description)
{
  fType = commandType;
  fDescription = description;
  fActive = false;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

CommandWithOption::CommandWithOption(Command::Type commandType, const G4String& description,
                                     const G4String& defaultOption, const G4String& optionName)
  : Command(commandType, description)
{
  fDefaultOption = defaultOption;
  fOptionName = optionName;
  fOption = "";
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

G4int CommandLineParser::Parse(int& argc, char** argv)
{
  static char null[1] = {""};
  G4int firstArgc = argc;

  for (G4int i = 1; i < firstArgc; i++) {
    auto command = FindCommand(argv[i]);
    if (command == nullptr) continue;

    if (fVerbose) G4cout << "Command : " << argv[i] << G4endl;

    fOptionsWereSetup = true;
    command->fActive = true;

    G4String marker(argv[i]);

    if (strcmp(argv[i], "-h") != 0 && strcmp(argv[i], "--help") != 0) {
      argv[i] = null;
    }

    if (command->fType == Command::WithOption) {
      if (fVerbose) G4cout << "WithOption" << G4endl;

      if (i + 1 > firstArgc || argv[i + 1] == nullptr || argv[i + 1][0] == '-') {
        G4cerr << "An command line option is missing for " << marker << G4endl;
        abort();
      }

      command->SetOption((const char*)strdup(argv[i + 1]));
      argv[i + 1] = null;
      i++;
    }
    else if (command->fType == Command::OptionNotCompulsory) {
      if (fVerbose) G4cout << "OptionNotCompulsory" << G4endl;

      if (i + 1 < firstArgc) {
        G4String buffer = (const char*)strdup(argv[i + 1]);

        if (!buffer.empty()) {
          if (buffer.at(0) != '-' && buffer.at(0) != '&' && buffer.at(0) != '>'
              && buffer.at(0) != '|')
          {
            if (fVerbose) {
              G4cout << "facultative option is : " << buffer << G4endl;
            }

            command->SetOption((const char*)strdup(argv[i + 1]));
            argv[i + 1] = null;
            i++;
            continue;
          }
        }
      }

      if (fVerbose) G4cout << "Option not set" << G4endl;

      command->SetOption("");
    }
  }
  CorrectRemainingOptions(argc, argv);

  Command* commandLine(nullptr);
  if ((commandLine = GetCommandIfActive("--help")) || (commandLine = GetCommandIfActive("-h"))) {
    G4cout << "Usage : " << argv[0] << " [OPTIONS]" << G4endl;
    PrintHelp();
    return 1;
  }

  return 0;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void CommandLineParser::PrintHelp()
{
  std::map<G4String, Command*>::iterator it;

  G4int maxFieldLength = fMaxMarkerLength + fMaxOptionNameLength + 4;

  G4cout << "Options: " << G4endl;

  for (it = fCommandMap.begin(); it != fCommandMap.end(); it++) {
    Command* command = it->second;
    if (command) {
      G4cout << setw(maxFieldLength) << left;

      G4String toPrint = it->first;

      if (toPrint == "&") {
        continue;
      }
      else if (toPrint == "-h")
        continue;
      else if (toPrint == "--help") {
        toPrint += ", -h";
      }

      if (command->GetDefaultOption() != "") {
        toPrint += " \"" + command->GetDefaultOption() + "\"";
      }
      G4cout << toPrint;
      G4cout << command->GetDescription() << G4endl;
    }
  }
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void CommandLineParser::CorrectRemainingOptions(int& argc, char** argv)
{
  // remove handled arguments from argument array
  G4int j = 0;
  for (G4int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "")) {
      argv[j] = argv[i];
      j++;
    }
  }
  argc = j;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void CommandLineParser::AddCommand(const G4String& marker, Command::Type type,
                                   const G4String& description, const G4String& defaultOption,
                                   const G4String& optionName)
{
  Command* command = nullptr;
  switch (type) {
    case Command::WithoutOption:
      command = new Command(type, description);
      break;

    default:
      command = new CommandWithOption(type, description, defaultOption, optionName);
      if ((int)defaultOption.length() > fMaxOptionNameLength)
        fMaxOptionNameLength = defaultOption.length();
      break;
  }

  if ((G4int)marker.length() > fMaxMarkerLength) fMaxMarkerLength = marker.length();
  fCommandMap.insert(make_pair(marker, command));
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

Command* CommandLineParser::FindCommand(const G4String& marker)
{
  auto it = fCommandMap.find(marker);
  if (it == fCommandMap.end()) {
    return nullptr;
  }
  return it->second;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

Command* CommandLineParser::GetCommandIfActive(const G4String& marker)
{
  Command* command = FindCommand(marker);
  if (command) {
    if (command->fActive) {
      return command;
    }
  }
  else {
    G4ExceptionDescription description;
    description << "You try to retrieve a command that was not registered : " << marker << G4endl;
    G4Exception("CommandLineParser::GetCommandIfActive", "COMMAND LINE NOT DEFINED", FatalException,
                description, "");
  }
  return nullptr;
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

bool CommandLineParser::CheckIfNotHandledOptionsExists(int& argc, char** argv)
{
  if (argc > 0) {
    G4bool kill = false;
    for (G4int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "")) {
        kill = true;
        G4cerr << "Unknown argument : " << argv[i] << "\n";
      }
    }
    if (kill) {
      G4cerr << "The option " << argv[0] << " is not handled this programme." << G4endl;
      G4cout << "Usage : " << argv[0] << " [OPTIONS]" << G4endl;
      PrintHelp();
      return true;  // KILL APPLICATION
    }
  }
  return false;
}
